from __future__ import annotations

import logging

from llm_knowledge_enhancement.system.candidate_retriever import retrieve_candidates
from llm_knowledge_enhancement.system.item_store import ItemStore
from llm_knowledge_enhancement.system.profile_builder import build_profile
from llm_knowledge_enhancement.system.ranker import rank
from llm_knowledge_enhancement.system.types import PurchaseEvent

DEFAULT_RECENT_HISTORY_SIZE = 6
DEFAULT_K = 3
DEFAULT_NEIGHBORS_PER_HISTORY_ITEM = 4
logger = logging.getLogger(__name__)


def run(
    user: str,
    user_purchase_history: list[PurchaseEvent],
    embedding_model: str,
    **kwargs,
) -> dict[str, object]:
    """Rank item-description neighbors for one user using the stored FAISS index.

    Wires UserProfileBuilder -> CandidateRetriever -> Ranker; see
    docs/architecture/recommendation-system.md.
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

    store = ItemStore(embedding_model)
    profile_item_ids = build_profile(user_purchase_history, recent_history_size)
    purchased_item_ids = {event.parent_asin for event in user_purchase_history}
    candidate_scores = retrieve_candidates(
        store, profile_item_ids, purchased_item_ids, neighbors_per_history_item
    )
    ranked_item_ids = rank(candidate_scores, k)

    return {
        "user_id": user,
        "predicted_item_ids": ranked_item_ids,
        "prediction_scores": {
            item_id: candidate_scores[item_id] for item_id in ranked_item_ids
        },
    }
