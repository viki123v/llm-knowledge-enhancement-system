# Prerequisites for the Evaluate module

The evaluate module is a pure function of its inputs. It needs exactly two things, and neither involves it reading a database.

## PR8. Embedding coverage for every id in the predictions file

The `EmbeddingMatrix` handed to `CosineEmbeddingScorer` must contain a row for **every** `true_item_id` and **every** entry of **every** `predicted_item_ids` list in the file being scored. A missing id means the evaluator cannot compute that item's relevance.

The run script builds this matrix from PR4 and injects it. Coverage should be asserted before scoring starts rather than discovered mid-run: with `on_missing="raise"` a gap surfaces as an exception partway through, which wastes the run.

- **Produces:** an in-memory `EmbeddingMatrix` (see [interfaces.md](interfaces.md#embeddingmatrix)).
- **Consumed by:** the evaluate module, via injection.

## PR9. Run manifest

A small JSON file written next to each predictions file recording what produced it: `run_id`, embedding `model_id` and revision, embedding dimension `d`, `K`, random `seed`, `split_version`, the description assembly version and truncation cap, and the timestamp.

Without this, two `metrics.json` files are not comparable, because you cannot tell whether a difference came from the ranker or from someone swapping the embedding model.

- **Produces:** `artifacts/predictions/<run_id>.manifest.json`.
- **Consumed by:** humans comparing runs, and by the results table.

## PR10. A valid predictions file

The System has already run and written `artifacts/predictions/<run_id>.jsonl` satisfying every rule in [interfaces.md](interfaces.md#predictionrecord-and-the-predictions-file): unique users, ranked order, no duplicates within a list, no history items, at least one prediction per user.

- **Consumed by:** the evaluate module, as its only file input.