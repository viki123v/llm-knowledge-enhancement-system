from __future__ import annotations

from llm_knowledge_enhancement.system.baseline import popularity_ranker
from llm_knowledge_enhancement.system.baseline.types import PurchaseEvent


def _event(item_id: str) -> PurchaseEvent:
    return PurchaseEvent(parent_asin=item_id, timestamp=0)


RANKING = (("a", 3), ("b", 1), ("c", 1))


def test_run_ranks_by_popularity_excluding_users_own_history():
    result = popularity_ranker.run(
        user="u1",
        user_purchase_history=[_event("b")],
        popularity_ranking=RANKING,
        k=2,
    )

    assert result["predicted_item_ids"] == ["a", "c"]
    assert result["prediction_scores"] == {"a": 3.0, "c": 1.0}


def test_run_truncates_to_k():
    result = popularity_ranker.run(
        user="u1",
        user_purchase_history=[],
        popularity_ranking=RANKING,
        k=1,
    )

    assert result["predicted_item_ids"] == ["a"]
