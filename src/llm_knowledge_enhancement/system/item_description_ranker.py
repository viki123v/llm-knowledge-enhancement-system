from __future__ import annotations

import logging

from llm_knowledge_enhancement.shared.item_embeddings import (
    load_item_embedding_index,
    reconstruct_vectors,
)
from llm_knowledge_enhancement.system.types import PurchaseEvent

DEFAULT_RECENT_HISTORY_SIZE = 6
DEFAULT_K = 3
DEFAULT_NEIGHBORS_PER_HISTORY_ITEM = 4
logger = logging.getLogger(__name__)


def _recent_purchase_ids(
    user_purchase_history: list[PurchaseEvent],
    recent_history_size: int,
) -> tuple[str, ...]:
    if recent_history_size < 1:
        raise ValueError("recent_history_size must be at least 1")
    if not user_purchase_history:
        raise ValueError("Cannot rank items for a user with an empty purchase history")

    recent_events = sorted(
        user_purchase_history,
        key=lambda event: (event.timestamp, event.parent_asin),
        reverse=True,
    )[:recent_history_size]
    return tuple(event.parent_asin for event in recent_events)


def run(
    user: str,
    user_purchase_history: list[PurchaseEvent],
    embedding_model: str,
    **kwargs,
) -> dict[str, object]:
    """Rank item-description neighbors for one user using the stored FAISS index.

    The item embeddings were written L2-normalized into an IndexFlatIP, so FAISS
    inner product is equivalent to cosine similarity. For a small, recency-bounded
    user profile, max similarity to any recent purchase preserves niche intent
    better than averaging all purchases into one vector.
    """
    k = int(kwargs.get("k", DEFAULT_K))
    recent_history_size = int(
        kwargs.get("recent_history_size", DEFAULT_RECENT_HISTORY_SIZE)
    )
    neighbors_per_history_item = int(
        kwargs.get(
            "neighbors_per_history_item",
            max(DEFAULT_NEIGHBORS_PER_HISTORY_ITEM, k + 1),
        )
    )
    if k < 1:
        raise ValueError("k must be at least 1")
    if neighbors_per_history_item < 1:
        raise ValueError("neighbors_per_history_item must be at least 1")

    item_index = load_item_embedding_index(embedding_model)
    recent_item_ids = _recent_purchase_ids(user_purchase_history, recent_history_size)
    query_vectors = reconstruct_vectors(item_index, recent_item_ids)
    purchased_item_ids = {event.parent_asin for event in user_purchase_history}

    search_k = min(item_index.index.ntotal, neighbors_per_history_item)
    scores, row_ids = item_index.index.search(query_vectors, search_k)

    candidate_scores: dict[str, float] = {}
    for query_scores, query_rows in zip(scores, row_ids, strict=True):
        for score, row_id in zip(query_scores, query_rows, strict=True):
            if row_id < 0:
                continue
            item_id = item_index.item_ids[int(row_id)]
            if item_id in purchased_item_ids:
                continue
            candidate_scores[item_id] = max(
                candidate_scores.get(item_id, -float("inf")), float(score)
            )

    ranked_item_ids = [
        item_id
        for item_id, _ in sorted(
            candidate_scores.items(),
            key=lambda item_score: (-item_score[1], item_score[0]),
        )[:k]
    ]

    return {
        "user_id": user,
        "predicted_item_ids": ranked_item_ids,
        "prediction_scores": {
            item_id: candidate_scores[item_id] for item_id in ranked_item_ids
        },
    }
