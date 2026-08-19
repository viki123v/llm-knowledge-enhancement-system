from __future__ import annotations

import logging

from llm_knowledge_enhancement.system.baseline.types import PurchaseEvent
from llm_knowledge_enhancement.system.baseline.utils import global_per_item_freq

DEFAULT_K = 3
logger = logging.getLogger(__name__)


def run(
    user: str,
    user_purchase_history: list[PurchaseEvent],
    **kwargs,
) -> dict[str, object]:
    """Recommend the globally most-purchased items, excluding the user's own history.

    Baseline to check whether item_description_ranker beats "recommend what
    everyone else buys"; see docs/architecture/recommendation-system.md.
    """
    k = int(kwargs.get("k", DEFAULT_K))
    seen = {event.parent_asin for event in user_purchase_history}

    predicted_item_ids: list[str] = []
    prediction_scores: dict[str, float] = {}
    for item_id, count in global_per_item_freq():
        if item_id in seen:
            continue
        predicted_item_ids.append(item_id)
        prediction_scores[item_id] = float(count)

        if len(predicted_item_ids) == k:
            break

    return {
        "user_id": user,
        "predicted_item_ids": predicted_item_ids,
        "prediction_scores": prediction_scores,
    }
