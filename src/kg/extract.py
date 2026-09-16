"""Pure functions turning raw item metadata and purchase artifacts into KG rows.

No file or database I/O here; callers load the raw dicts (from the metadata
file and the existing `ingest/simple` purchase artifacts) and these functions
shape them into the rows `kg.store.KnowledgeGraphStore` writes.
"""

from __future__ import annotations

from llm_knowledge_enhancement.system.baseline.types import UserPurchaseHistory


def item_row(meta: dict) -> dict:
    """One `Item` node's properties from a raw metadata row."""
    return {
        "item_id": meta["parent_asin"],
        "title": meta.get("title") or "",
        "price": _as_float(meta.get("price")),
        "main_category": meta.get("main_category") or "",
    }


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def category_edges(meta: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """`(item -> category)` and `(child category -> parent category)` edges."""
    categories = [c for c in (meta.get("categories") or []) if c]
    item_id = meta["parent_asin"]
    in_category = [(item_id, category) for category in categories]
    subcategory_of = list(zip(categories[1:], categories[:-1]))
    return in_category, subcategory_of


def brand_edge(meta: dict) -> tuple[str, str] | None:
    """`(item -> brand)`, or `None` when the item has no brand (58% coverage)."""
    brand = (meta.get("details") or {}).get("Brand")
    return (meta["parent_asin"], brand) if brand else None


def store_edge(meta: dict) -> tuple[str, str] | None:
    """`(item -> store)`, or `None` when the item has no store."""
    store = meta.get("store")
    return (meta["parent_asin"], store) if store else None


def purchase_edges_from_history(
    history: UserPurchaseHistory,
) -> list[tuple[str, str, int]]:
    """`(user -> item, timestamp)` edges from training purchase history."""
    return [
        (user_id, event.parent_asin, event.timestamp)
        for user_id, events in history.items()
        for event in events
    ]


def purchase_edges_from_held_out(held_out: list[dict]) -> list[tuple[str, str, int]]:
    """`(user -> item, timestamp)` edges from the held-out next purchase."""
    return [
        (row["user_id"], row["parent_asin"], row["timestamp"]) for row in held_out
    ]
