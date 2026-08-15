from __future__ import annotations

from llm_knowledge_enhancement.system.baseline.types import PurchaseEvent


def build_profile(
    user_purchase_history: list[PurchaseEvent],
    recent_history_size: int,
) -> tuple[str, ...]:
    """Select the most recent item ids to search from, most recent first."""
    if recent_history_size < 1:
        raise ValueError("recent_history_size must be at least 1")
    if not user_purchase_history:
        raise ValueError("Cannot build a profile for a user with an empty purchase history")

    recent_events = sorted(
        user_purchase_history,
        key=lambda event: (event.timestamp, event.parent_asin),
        reverse=True,
    )[:recent_history_size]

    return tuple(event.parent_asin for event in recent_events)
