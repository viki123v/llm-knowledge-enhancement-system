from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import ijson

from llm_knowledge_enhancement.paths import REPO_ROOT
from llm_knowledge_enhancement.system.baseline.types import (
    PurchaseEvent,
    UserPurchaseHistory,
)

USER_PURCHASE_HISTORY_PATH = (
    REPO_ROOT / "data" / "processed" / "simple" / "user_purchase_history.json"
)


def iter_user_purchase_history(
    path: Path = USER_PURCHASE_HISTORY_PATH,
) -> Iterator[tuple[str, list[PurchaseEvent]]]:
    """Stream `(user_id, purchase_history)` pairs without loading the full file."""
    with path.open("rb") as f:
        for user_id, history in ijson.kvitems(f, ""):
            yield str(user_id), [PurchaseEvent.from_dict(event) for event in history]


def load_user_purchase_history(
    path: Path = USER_PURCHASE_HISTORY_PATH,
) -> UserPurchaseHistory:
    return dict(iter_user_purchase_history(path))


@lru_cache(maxsize=1)
def global_per_item_freq() -> tuple[tuple[str, int], ...]:
    """All item ids ranked by purchase count across all users, most popular first."""
    counts: Counter[str] = Counter()
    for history in load_user_purchase_history().values():
        counts.update(event.parent_asin for event in history)
    return tuple(sorted(counts.items(), key=lambda item_count: (-item_count[1], item_count[0])))
