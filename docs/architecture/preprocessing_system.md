# Prerequisites for the System

## PR6. Chronological leave-one-out split

For each eligible user, sort positive interactions by `timestamp_ms` with deterministic tie-breaking, then:

- the **latest** positive becomes the held-out true next purchase,
- all **earlier** positives become the training history.

A user is eligible when they have at least two positives (one to learn from, one to hold out) and the held-out item exists in the catalog. Users failing either condition are excluded, and the excluded count is reported rather than hidden.

Random splits are not acceptable here: predicting the *next* purchase from the past is a temporal question, and shuffling makes future purchases visible as history.

- **Produces:** `artifacts/split/<split_version>/train_history.parquet` and `held_out.parquet`.
- **Consumed by:** the System (history only) and the run script (held-out item, to fill `true_item_id`).

## PR7. Per-user mask set

For each user, the set of every item they have already interacted with, at **any** rating, not only positives. The ranker removes these from its candidate list, because recommending something already bought is not a prediction of the next purchase. Low-rated items are still masked: the user has seen them.

- **Produces:** derivable from PR5; materialize as a `user_id -> set[item_id]` mapping for speed.
- **Consumed by:** `CandidateRetriever` and `Ranker`.
