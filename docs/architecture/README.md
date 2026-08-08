# Next-Purchase Recommendation and Evaluation Architecture

This directory specifies two independent halves of the project:

1. **The System** — given a user's past purchases, produce a ranked list of items, most likely next purchase first, by matching item descriptions against what the user has already bought.
2. **The Evaluate module** — given the one true next-purchased item and the ranked list the system produced, compute NDCG@10 and a top-1 exact-match boolean.

They are deliberately kept apart. The evaluate module is a separate phase so that it can be written, tested, and changed without touching the recommender, and so that a completely different recommender can be dropped in later and scored by the same code.

## The decoupling rule

> The System never imports the Evaluate module. The Evaluate module never imports the System, never opens a database connection, and never reads project configuration.

They meet at exactly one place: a **predictions file** (`predictions.jsonl`). The system writes it; the evaluator reads it. Everything the evaluator needs beyond that file — the item embeddings used to measure semantic relevance — is *passed in by the caller*, never fetched by the evaluator itself. This is the single constraint that keeps the two phases from growing into each other.

## Component map

```mermaid
flowchart LR
  subgraph prep [Preprocessing]
    itemTable[Canonical item table]
    interactions[Interaction table]
    splitter[Chronological leave-one-out split]
    embedder[Frozen embedding model]
  end
  subgraph store [Item Store]
    pgvector[(descriptions + vectors)]
  end
  subgraph system [Recommendation System]
    profile[UserProfileBuilder]
    retriever[CandidateRetriever]
    ranker[Ranker]
    writer[PredictionWriter]
  end
  predfile[[predictions.jsonl]]
  subgraph evalmod [Evaluation Module]
    scorer[RelevanceScorer]
    metrics[NDCG@10 and top-1 match]
    report[Aggregated report]
  end
  itemTable --> embedder --> pgvector
  interactions --> splitter
  splitter -->|train history| profile
  splitter -->|held-out true item| writer
  pgvector --> profile
  pgvector --> retriever
  profile --> retriever --> ranker --> writer --> predfile
  predfile --> metrics
  pgvector -.->|embedding matrix handed in by the caller| scorer
  scorer --> metrics --> report
```

Solid arrows are hard dependencies. The dotted arrow is the one place the evaluator touches store-derived data, and it does so only through an object the caller constructs and injects.

## Lifecycle

A full run is three sequential phases. Each phase finishes completely and writes an artifact before the next begins; nothing streams between them.

| Phase | Runs | Reads | Writes |
| --- | --- | --- | --- |
| Preprocess | once per data version | raw review and metadata files | canonical item table, interaction table, split, item embeddings, loaded item store |
| Predict | once per experiment run | item store, per-user train history | `artifacts/predictions/<run_id>.jsonl` |
| Evaluate | once per predictions file, repeatable | `predictions.jsonl` plus an injected embedding matrix | `results/<run_id>.metrics.json` and an optional per-user CSV |

Because evaluation only consumes a file, you can re-score an old run after changing a metric without re-running prediction, and you can score a competitor's predictions file that was produced by code this repository has never seen.

## Metrics at a glance

Two numbers per run, both defined precisely in [evaluation.md](evaluation.md):

- **NDCG@10** with *graded* relevance. The relevance of a predicted item is its semantic similarity to the true item, not a 0/1 hit. The ideal DCG is the same ten similarities sorted descending.
- **Top-1 exact match** (the "True/False" metric). `True` when the rank-1 prediction *is* the true item.

These two measure different things and must be read together. NDCG@10 as defined here measures ordering quality *within* the returned list; it says nothing about whether the list contains anything correct. Top-1 exact match is the metric that carries correctness. See the caveat section in [evaluation.md](evaluation.md) before quoting NDCG@10 on its own.

## Document index

| Document | What it specifies |
| --- | --- |
| [interfaces.md](interfaces.md) | Every shared data structure and `Protocol`. Read this first; the other docs reference its names rather than redefining them. |
| [preprocessing.md](preprocessing.md) | The `Preprocessing` section: all prerequisites both halves assume, with the artifact each produces and which component consumes it. |
| [item-store.md](item-store.md) | The separate store holding item descriptions and vectors, its interface, indexing, and test backend. |
| [recommendation-system.md](recommendation-system.md) | `UserProfileBuilder`, `CandidateRetriever`, `Ranker`, `PredictionWriter`: responsibilities and guidelines. |
| [evaluation.md](evaluation.md) | The `src/evaluation` package: metric formulas, layout, `main.py` CLI contract, and the NDCG caveat. |

## Conventions used across these documents

- `item_id` is the canonical item key and equals `parent_asin`, never the variant `asin`. See [table_schema.md](../../table_schema.md).
- `K` is the length of the returned ranked list. Default `K = 10`, matching the reported cutoff.
- Every component section follows the same shape: responsibility, inputs, outputs, the protocol it implements, what it must not do, and failure modes.
- Type signatures are written as Python 3.13 with `typing.Protocol`. They are contracts, not committed code; implementations live under `src/`.
