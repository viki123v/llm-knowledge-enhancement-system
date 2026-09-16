from __future__ import annotations

import pytest

from kg import tools
from kg.store import KGNode


class StubStore:
    """In-memory stand-in for KnowledgeGraphStore; no Neo4j needed."""

    def get_node(self, node_id: str) -> KGNode | None:
        if node_id != "B08BBQ29N5":
            return None
        return KGNode(id=node_id, label="Item", properties={"title": "Electric Guitar"})

    def neighbors(self, node_id: str, relation: str | None = None) -> list[KGNode]:
        assert node_id == "B08BBQ29N5"
        if relation == "HAS_BRAND":
            return [KGNode(id="Fender", label="Brand", properties={})]
        return []

    def items_by_attribute(self, *, brand=None, category=None, store=None) -> list[str]:
        return ["B08BBQ29N5"] if brand == "Fender" else []

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        return [source_id, target_id] if source_id != target_id else None


def test_dispatch_get_node_found_and_missing():
    store = StubStore()
    assert tools.dispatch(store, "kg_get_node", {"node_id": "B08BBQ29N5"}) == {
        "id": "B08BBQ29N5",
        "label": "Item",
        "title": "Electric Guitar",
    }
    assert tools.dispatch(store, "kg_get_node", {"node_id": "unknown"}) is None


def test_dispatch_neighbors_filters_by_relation():
    store = StubStore()
    result = tools.dispatch(
        store, "kg_neighbors", {"node_id": "B08BBQ29N5", "relation": "HAS_BRAND"}
    )
    assert result == [{"id": "Fender", "label": "Brand"}]


def test_dispatch_items_by_attribute():
    store = StubStore()
    assert tools.dispatch(store, "kg_items_by_attribute", {"brand": "Fender"}) == [
        "B08BBQ29N5"
    ]


def test_dispatch_shortest_path():
    store = StubStore()
    assert tools.dispatch(
        store, "kg_shortest_path", {"source_id": "a", "target_id": "b"}
    ) == ["a", "b"]


def test_dispatch_unknown_tool_raises():
    with pytest.raises(ValueError):
        tools.dispatch(StubStore(), "not_a_tool", {})


def test_every_tool_schema_has_a_dispatch_branch():
    schema_names = {tool["function"]["name"] for tool in tools.TOOLS}
    assert schema_names == {
        "kg_get_node",
        "kg_neighbors",
        "kg_items_by_attribute",
        "kg_shortest_path",
    }
