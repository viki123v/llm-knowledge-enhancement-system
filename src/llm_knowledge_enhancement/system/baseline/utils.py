from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import ijson

from llm_knowledge_enhancement.system.baseline.types import (
    PurchaseEvent,
    UserPurchaseHistory,
)
from shared.paths import USER_PURCHASE_HISTORY_PATH


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
