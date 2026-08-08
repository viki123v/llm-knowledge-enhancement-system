# Evaluation Module

A separate phase that scores a ranked list against the one item the user actually bought next. It is deliberately isolated: no database, no network, no imports from the rest of the project. Give it a predictions file and a way to measure similarity between two item ids, and it returns numbers.

## Responsibility

For each user with a true next-purchased item `t` and a ranked prediction list `p_1 .. p_K`, compute:

1. **NDCG@10** with graded relevance, where relevance is the semantic similarity between `t` and each predicted item.
2. **Top-1 exact match** — the True/False metric. `True` when `p_1 == t`.

Then aggregate across users and emit a report.

## Metric definitions

Let `t` be the true item and `p_1 .. p_K` the predictions in rank order, `i` starting at 1.

### Relevance

```
rel_i = clamp(sim(t, p_i), 0, 1)
```

`sim` comes from the injected [`RelevanceScorer`](interfaces.md#relevancescorer). The default is cosine similarity between item embeddings. Cosine can be negative; negatives clamp to `0` so that discounted gains stay non-negative and DCG remains monotone in relevance. An exact match contributes `rel = 1.0`, since an item's cosine similarity with itself is 1.

### NDCG@K

```
DCG@K  = sum_{i=1..K}  rel_i / log2(i + 1)
IDCG@K = sum_{i=1..K}  sorted_desc(rel)_i / log2(i + 1)
NDCG@K = DCG@K / IDCG@K            (defined as 0.0 when IDCG@K == 0)
```

The ideal ranking is **the same K relevance values, sorted descending**. So NDCG@K asks: did the ranker put its most semantically relevant candidates at the top of the list it returned?

`IDCG@K == 0` happens only when every predicted item has zero relevance to the true item. The report counts these users separately rather than folding a defined-by-convention `0.0` into the mean without trace.

### Top-1 exact match (the True/False metric)

```
TopOneMatch = (p_1 == t)
```

A plain boolean per user. Aggregated as a mean, it is top-1 accuracy: the fraction of users whose very next purchase the system put first.

### Aggregation

Both metrics are averaged unweighted across evaluated users. Every user counts once regardless of history length, since the question is per-user prediction quality, not per-interaction.

## The NDCG caveat, stated plainly

**NDCG@10 as defined here measures ordering quality within the returned list, not whether the list is any good.**

Because the ideal DCG is built from the same ten similarities the system returned, the metric normalizes away the absolute quality of the candidates. A system that returns ten items with nothing to do with the true item still scores near 1.0 if it happens to order those ten by descending similarity. Conversely a system that returns the true item at rank 2 and nine perfect near-duplicates scores below 1.0 despite being excellent.

Consequences for reading results:

- **Never quote NDCG@10 alone.** Always report it beside top-1 exact match, which carries whether the prediction was actually correct.
- NDCG@10 is a diagnostic for the ranking stage: it isolates "given these candidates, did you order them well?" from "were these the right candidates?"
- Two runs with different candidate generation are not comparable on NDCG@10 in the way they would be with catalog-normalized NDCG, because each is normalized against its own candidates.
- The report includes `mean_relevance@10`, the unnormalized mean of `rel_i`, precisely so the absolute quality that NDCG divides out is still visible. If NDCG is high while mean relevance is low, the ranker is ordering junk neatly.

This normalization was chosen deliberately for its cheapness — it needs no catalog-wide pass per user. The trade-off is real and the report is designed to keep it visible rather than hidden.

## Package layout

```
src/evaluation/
├── __init__.py        # public API re-exports: evaluate, EvaluationReport, CosineEmbeddingScorer
├── main.py            # CLI entry point
├── contracts.py       # PredictionRecord, EmbeddingMatrix, RelevanceScorer protocol
├── io.py              # read_predictions(), write_report()
├── scorers.py         # CosineEmbeddingScorer and any alternative scorers
├── metrics.py         # ndcg_at_k(), top_one_match(), dcg()
└── report.py          # UserResult, EvaluationReport, aggregation
```

A package rather than one file, because there are four separable concerns — parsing, similarity, metric math, aggregation — and `metrics.py` in particular should be readable and testable with no knowledge of files or embeddings. `metrics.py` takes floats and returns floats; that is the piece worth keeping pristine.

### Dependencies

`numpy` only, plus the standard library. No `pandas`, no `psycopg2`, no `duckdb`, and nothing from `src/` outside this package. If an import of a project module ever appears here, the decoupling has been broken.

### Packaging note

The project's `pyproject.toml` uses the `uv_build` backend with the package at `src/llm_knowledge_enhancement`. Adding `src/evaluation` as a sibling top-level package requires the build backend to be told about it (`[tool.uv.build-backend]` module configuration, or a namespace layout). Worth handling when the package is first created rather than discovering it at install time.

## Public API

```python
def evaluate(
    records: Iterable[PredictionRecord],
    scorer: RelevanceScorer,
    *,
    k: int = 10,
) -> EvaluationReport: ...
```

Pure: same inputs, same outputs, no I/O, no globals, no clock. This is the function tests exercise, and it is what makes the module reusable from a notebook.

```python
@dataclass(frozen=True, slots=True)
class UserResult:
    user_id: str
    ndcg_at_k: float
    top_one_match: bool
    mean_relevance: float
    n_predictions: int          # actual list length, may be < k
    undefined_idcg: bool        # every relevance was zero

@dataclass(frozen=True, slots=True)
class EvaluationReport:
    k: int
    n_users: int
    mean_ndcg_at_k: float
    top_one_accuracy: float
    mean_relevance_at_k: float
    n_undefined_idcg: int
    n_short_lists: int          # users with fewer than k predictions
    per_user: tuple[UserResult, ...]
```

Keeping `per_user` on the report means significance testing, error analysis, and slicing by history length can all happen later without re-running evaluation.

### Typical use

```python
from evaluation import evaluate, CosineEmbeddingScorer
from evaluation.io import read_predictions

matrix = build_embedding_matrix()            # caller's job, from artifacts
scorer = CosineEmbeddingScorer(matrix)
report = evaluate(read_predictions(path), scorer, k=10)

print(report.mean_ndcg_at_k, report.top_one_accuracy)
```

The caller constructs the matrix. The module never does. That single line of separation is what keeps the evaluator free of the store, the config, and the database.

## Edge cases and how they are handled

| Case | Behavior |
| --- | --- |
| Prediction list shorter than `k` | Score what is there. DCG and IDCG both run over the actual length, so a short list is not penalized by the normalization; `n_short_lists` records it. |
| Prediction list empty | Reject at parse time. An empty list is a malformed record, not a zero score. |
| All relevances zero | `NDCG = 0.0`, flagged via `undefined_idcg`, counted in `n_undefined_idcg`. |
| True item appears at rank 1 | `rel_1 = 1.0` and `TopOneMatch = True`. |
| True item present but not at rank 1 | Contributes `rel = 1.0` at its position; `TopOneMatch = False`. |
| Duplicate `user_id` in the file | Reject at parse time. Duplicates would double-count in the mean. |
| Duplicate item within one prediction list | Reject at parse time, per the file contract. |
| Item id missing from the embedding matrix | `on_missing="raise"` (default) raises; `on_missing="zero"` treats relevance as `0.0`. Default to raising in real runs — a silent zero looks like a bad prediction rather than a broken setup. |
| `k` larger than every list | Legal; every user is a short list. |

## CLI contract

`main.py` is a thin shell around `evaluate`: parse arguments, load, call, write. No metric logic lives in it.

```
python -m evaluation.main \
    --predictions artifacts/predictions/<run_id>.jsonl \
    --embeddings  artifacts/embeddings/<model_id>/items.npy \
    --embedding-ids artifacts/embeddings/<model_id>/item_ids.json \
    --k 10 \
    --output results/<run_id>.metrics.json \
    [--per-user-csv results/<run_id>.per_user.csv] \
    [--on-missing raise|zero]
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `--predictions` | yes | JSONL file produced by the system |
| `--embeddings` | yes | `.npy` float32 matrix, rows aligned to `--embedding-ids` |
| `--embedding-ids` | yes | JSON array of item ids, one per matrix row |
| `--k` | no, default 10 | Cutoff |
| `--output` | yes | Where the JSON report is written |
| `--per-user-csv` | no | Also dump per-user rows for error analysis |
| `--on-missing` | no, default `raise` | Behavior for ids absent from the matrix |

Loading embeddings from files at the CLI boundary does not violate the decoupling rule: `main.py` is the caller, and it constructs the `EmbeddingMatrix` that gets injected. The library code beneath it still touches no files. Exit code is `0` on success and non-zero on a contract violation, so a pipeline fails loudly rather than producing an empty report.

### Report output

```json
{
  "k": 10,
  "n_users": 52431,
  "mean_ndcg_at_k": 0.7412,
  "top_one_accuracy": 0.0183,
  "mean_relevance_at_k": 0.3126,
  "n_undefined_idcg": 12,
  "n_short_lists": 47,
  "predictions_file": "artifacts/predictions/run_007.jsonl",
  "embedding_model_id": "<copied from the run manifest>"
}
```

The embedding model id is copied through from the run manifest ([PR9](preprocessing.md#pr9-run-manifest)) so a report can always be traced to the space its similarities were measured in.

## Testing guidelines

The module is pure, so it can be tested exhaustively with no fixtures beyond a stub scorer.

- **Metric math against hand-computed values.** With relevances `[1.0, 0.5, 0.0]`, `DCG = 1/1 + 0.5/1.585 + 0 = 1.3155`. Since these are already sorted descending, `IDCG` is equal and `NDCG = 1.0`. Reverse the list and NDCG must drop.
- **Perfect ordering scores exactly 1.0**; reversed ordering scores strictly less; a constant relevance list scores exactly 1.0 regardless of order, which is worth an explicit test since it is a direct expression of the caveat above.
- **Top-1 match is position-sensitive.** Moving the true item from rank 1 to rank 2 must flip it to `False` while leaving DCG's true-item contribution merely discounted.
- **Stub scorer** returning fixed values from a dict, to test metrics with no embeddings at all.
- **Contract violations reject.** Duplicate users, duplicate items, empty lists, and unknown ids under `on_missing="raise"` each produce a clear error naming the offending user.
- **Determinism.** The same file and matrix produce byte-identical reports across runs.
