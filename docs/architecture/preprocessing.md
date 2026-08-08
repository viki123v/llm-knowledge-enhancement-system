# Preprocessing

Everything the System and the Evaluate module take for granted. These are prerequisites, not part of either module: both assume the artifacts below already exist and are correct. Each entry names what it produces, who consumes it, and the checks that must pass before the artifact is considered done.

Preprocessing runs once per data version. Prediction and evaluation may then run many times against the same frozen artifacts.

## Contents

- [Prerequisites for both halves](#prerequisites-for-both-halves)
- [Validation gate](#validation-gate)

# Prerequisites for both halves

### PR1. Canonical item table

Every item in the catalog, keyed by `item_id = parent_asin`. The variant `asin` is never the key; multiple variants collapse into one product family. Uniqueness of `parent_asin` must be asserted, not assumed.

- **Produces:** `artifacts/canonical/items.parquet` with columns `item_id`, `description_text`, plus any audit columns kept for inspection.
- **Consumed by:** the item store (PR3), which loads it.

### PR2. Assembled item description text

One string per item, built once, so that the system and the evaluator can never disagree about what an item's description is. This is the field the whole approach rests on: the system finds relevant items by how well descriptions match, and the evaluator's relevance signal is the similarity between descriptions.

Assembly rules:

- Concatenate, in a fixed documented order: `title`, `categories` (ordered path), brand from `details.Brand` when present, `features` bullets, `description` fragments.
- Deduplicate exact-duplicate bullets and fragments; `features` and `description` frequently repeat each other.
- Normalize whitespace only. Do not lowercase, stem, or otherwise change meaning; the embedding model handles that.
- Apply a fixed character or token cap and record it. Truncation must be deterministic.
- **Exclude `average_rating` and `rating_number`.** Per [table_schema.md](../../table_schema.md) these are crawl-time aggregates that postdate the historical interactions and leak the target. They must not reach the text, the embeddings, or any model input.
- Items whose assembled text is empty after normalization are dropped from the catalog and the drop count is recorded.

- **Produces:** the `description_text` column of PR1.
- **Consumed by:** the embedding pass (PR4), and anything that needs to show a human why an item was recommended.

### PR3. Item store loaded and indexed

The store described in [item-store.md](item-store.md), populated from PR1 and PR4, covering **every** catalog item. "All the items in the given store" is the candidate universe, so an item missing from the store is an item the system can never recommend.

- **Produces:** a queryable store (pgvector table plus vector index, or the parquet-backed fallback).
- **Consumed by:** the System, for both profile building and candidate retrieval. Also read by the *run script* to build the evaluator's embedding matrix — never by the evaluator itself.

### PR4. Item embeddings from one frozen model

Every catalog item embedded by a single, version-pinned, frozen embedding model. Vectors are float32, fixed dimension `d`, L2-normalized at write time so downstream cosine similarity is a dot product.

**This is the prerequisite most likely to be violated silently.** The system ranks with these vectors and the evaluator measures relevance with these vectors. If the two ever use different models, different revisions of the same model, or differently normalized vectors, the NDCG@10 numbers are not wrong in an obvious way — they are quietly meaningless. Pin the model id and revision, hash the assembled text, and record both in the run manifest (PR9).

- **Produces:** `artifacts/embeddings/<model_id>/items.npy` plus a parallel `item_ids.json`, and the vector column in the store.
- **Consumed by:** the item store (PR3) and the `EmbeddingMatrix` handed to the evaluator (PR8).

### PR5. Canonical interaction table

One row per observed user–item event, deduplicated to the **earliest** event per `(user_id, item_id)` pair, matching the official 0-core rule. Fields: `user_id`, `item_id`, `rating`, `timestamp_ms`, and derived `is_positive = rating >= 4`.

Treating a repeat interaction as a new event would let the same item be both history and target, so deduplication happens before splitting, not after.

- **Produces:** `artifacts/canonical/interactions.parquet`.
- **Consumed by:** the split (PR6).

# Validation gate

Preprocessing is only complete when all of these pass. Each is cheap and each catches a failure mode that is expensive to notice later.

| Check | Fails when |
| --- | --- |
| `parent_asin` is unique in the item table | Variant rows leaked in as separate items |
| Every interaction's `item_id` joins to an item | Split will produce targets the store cannot serve |
| No `(user_id, item_id)` pair appears twice in interactions | Deduplication was skipped |
| Every user's history is strictly increasing in time | Tie-breaking is nondeterministic |
| No user's history contains their held-out item | Leakage; every downstream number is invalid |
| Embedding count equals catalog size and all norms are 1 within tolerance | Partial or unnormalized embedding pass |
| `average_rating` and `rating_number` appear in no assembled text | Target leakage into descriptions |
| Manifest `model_id` matches the model that produced the stored vectors | System and evaluator are in different embedding spaces |
