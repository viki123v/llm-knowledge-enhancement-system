# Item Store

The separate place where items live with their descriptions, so the system always knows where to find the information it ranks on. It is the candidate universe: **every item in the store is a possible recommendation, and nothing outside it can ever be recommended.**

## Responsibility

Own item descriptions and their vectors, and answer three kinds of question:

1. What is this item? (`get`, `get_many`)
2. Give me the vectors for these items. (`embeddings_for`)
3. Which items are most similar to this vector? (`search`)

That is the whole job. The store holds no user data, no interaction history, and no notion of who is being recommended to.

## Interface

Implements the `ItemStore` protocol from [interfaces.md](interfaces.md#itemstore):

```python
class ItemStore(Protocol):
    def get(self, item_id: str, *, with_embedding: bool = True) -> ItemRecord | None: ...
    def get_many(self, item_ids: Sequence[str], *, with_embedding: bool = True) -> list[ItemRecord]: ...
    def embeddings_for(self, item_ids: Sequence[str]) -> np.ndarray: ...
    def search(self, query: np.ndarray, *, top_n: int,
               exclude_item_ids: Collection[str] = ()) -> list[ScoredItem]: ...
    def all_item_ids(self) -> Sequence[str]: ...
    def __len__(self) -> int: ...
```

### Method contracts

| Method | Returns | On unknown id |
| --- | --- | --- |
| `get` | one `ItemRecord`, or `None` | `None` |
| `get_many` | records for the ids that exist, order unspecified | silently omitted, so check `len` |
| `embeddings_for` | `(len(item_ids), d)` float32, rows in the order requested | raises `KeyError` |
| `search` | up to `top_n` `ScoredItem`, descending score | n/a |
| `all_item_ids` | every id in the store | n/a |

`get_many` is lenient because callers often probe optimistically. `embeddings_for` is strict because a missing vector at that point is always a bug, and returning a zero row would corrupt a user profile without anyone noticing.

`search` scores by cosine similarity. Since all stored vectors are L2-normalized (see [preprocessing.md](preprocessing.md#pr4-item-embeddings-from-one-frozen-model)), this is a dot product. `exclude_item_ids` is applied by the store, not by the caller after the fact: filtering afterwards would silently shrink a `top_n` result below what the caller asked for.

## Backends

Two implementations behind the same protocol. Nothing above the store may branch on which one is active.

### Primary: pgvector

Uses the `vector-db` service already defined in [compose.yml](../../compose.yml) (`pgvector/pgvector:pg18-trixie`).

```sql
CREATE TABLE items (
    item_id          text PRIMARY KEY,   -- parent_asin
    description_text text NOT NULL,
    embedding        vector(384) NOT NULL
);

CREATE INDEX items_embedding_hnsw
    ON items USING hnsw (embedding vector_ip_ops);
```

Notes:

- The dimension in `vector(384)` must match the frozen embedding model's output. Change the model, change the column, re-embed everything. Do not mix dimensions in one table.
- `vector_ip_ops` (inner product) is correct **because vectors are normalized**. With normalized vectors inner product ranks identically to cosine. If normalization is ever dropped, switch to `vector_cosine_ops` or the ranking silently becomes length-biased.
- HNSW is approximate. It is fine for candidate generation, where the goal is a good shortlist, and it is why the ranker rescores exactly afterwards.
- Exclusions go into the query (`WHERE item_id <> ALL($2)`) with an over-fetch margin, so the store still returns `top_n` after filtering.

### Fallback: in-memory / parquet

Loads `artifacts/canonical/items.parquet` and `artifacts/embeddings/<model_id>/items.npy` into a dict and a NumPy matrix, and implements `search` as one exact matrix multiply plus a partial sort.

This is not a toy. At the 5-core scale of roughly 25k items and 384 dimensions the matrix is about 38 MB and an exact search over the whole catalog takes milliseconds, which is faster end to end than round-tripping to Postgres. Use it for tests, notebooks, and any run where exactness matters more than persistence; use pgvector when the catalog grows or when the store must be shared across processes.

Both backends must produce identical `get` and `embeddings_for` results for the same data. `search` may differ in the tail because HNSW is approximate; this is the only permitted divergence and it should be verified as a recall@top_n comparison, not assumed.

## Guidelines

**The store owns all description and vector access.** No other component reads `items.parquet`, opens a database connection, or loads the embedding matrix for ranking purposes. When a component needs an item's text or vector it asks the store. This is what makes the backend swap above possible.

**Descriptions arrive pre-assembled.** The store does not concatenate title, features, and categories; that happened once in preprocessing (PR2). The store persists a string and hands it back unchanged.

**The store is read-only during prediction and evaluation.** It is written once, at load time. A run that mutates the store mid-experiment cannot be reproduced.

**Load is all-or-nothing.** A partially loaded store looks healthy and quietly shrinks the candidate universe. Load into a staging table and swap, or verify `len(store) == expected_catalog_size` before the store is considered ready.

## What it must not do

- Must not know about users, histories, splits, or held-out items.
- Must not compute embeddings. It stores what the embedding pass produced.
- Must not be reached by the evaluation module. The run script may read the store to build an `EmbeddingMatrix`, but the evaluator receives that matrix as an argument and never holds a store reference. See the decoupling rule in [README.md](README.md#the-decoupling-rule).
- Must not apply per-user business rules. Exclusion lists are passed in; the store does not decide what to exclude.

## Failure modes

| Symptom | Likely cause | Guard |
| --- | --- | --- |
| `search` returns fewer than `top_n` | exclusions applied without over-fetch, or store smaller than expected | over-fetch by the exclusion count; assert store size at load |
| Recommendations look random | vectors written unnormalized while using inner-product ops | assert all norms are 1 within tolerance at load |
| `KeyError` from `embeddings_for` | embedding pass covered fewer items than the item table | compare counts in the validation gate |
| pgvector and fallback disagree beyond the tail | different embedding artifact loaded into each | compare `model_id` from the manifest at load |
| Dimension mismatch error on insert | model swapped without recreating the column | make `d` part of the table's migration, keyed by `model_id` |
