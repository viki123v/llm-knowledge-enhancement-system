from __future__ import annotations


def rank(candidate_scores: dict[str, float], k: int) -> list[str]:
    """Sort candidates by score descending, tie-break by item id, truncate to k."""
    if k < 1:
        raise ValueError("k must be at least 1")

    return [
        item_id
        for item_id, _ in sorted(
            candidate_scores.items(),
            key=lambda item_score: (-item_score[1], item_score[0]),
        )[:k]
    ]
