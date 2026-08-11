from __future__ import annotations

from llm_knowledge_enhancement.system.item_store import ItemStore


def retrieve_candidates(
    store: ItemStore,
    profile_item_ids: tuple[str, ...],
    purchased_item_ids: set[str],
    neighbors_per_history_item: int,
) -> dict[str, float]:
    """Search neighbors of each profile item and merge by max score.

    Excludes anything in purchased_item_ids. A candidate found near more
    than one profile item keeps its highest score, preserving niche intent
    that averaging the profile items would wash out.
    """
    if neighbors_per_history_item < 1:
        raise ValueError("neighbors_per_history_item must be at least 1")

    query_vectors = store.get_vectors(profile_item_ids)
    scores, row_ids = store.search(query_vectors, neighbors_per_history_item)

    candidate_scores: dict[str, float] = {}
    for query_scores, query_rows in zip(scores, row_ids, strict=True):
        for score, row_id in zip(query_scores, query_rows, strict=True):
            if row_id < 0:
                continue
            item_id = store.item_id_at(int(row_id))
            if item_id in purchased_item_ids:
                continue
            candidate_scores[item_id] = max(
                candidate_scores.get(item_id, -float("inf")), float(score)
            )
    return candidate_scores
