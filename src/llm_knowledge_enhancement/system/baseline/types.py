from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PurchaseEvent:
    parent_asin: str
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PurchaseEvent:
        return cls(
            parent_asin=str(data["parent_asin"]),
            timestamp=int(data["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class UserItem:
    """Held-out last purchase for a user (`user_item.json`)."""

    user_id: str
    parent_asin: str
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserItem:
        return cls(
            user_id=str(data["user_id"]),
            parent_asin=str(data["parent_asin"]),
            timestamp=int(data["timestamp"]),
        )


# user_purchase_history.json: {user_id: [PurchaseEvent, ...]}
UserPurchaseHistory = dict[str, list[PurchaseEvent]]


class SystemModel(ABC):
    def run(self) -> None:
        pass
