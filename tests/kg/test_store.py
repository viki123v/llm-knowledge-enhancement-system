"""Integration test against a real Neo4j instance (the `graph-db` compose
service). Skipped entirely when it isn't reachable, per the design spec's
testing section.
"""

from __future__ import annotations

import os
import uuid

import pytest

from kg.store import KnowledgeGraphStore


@pytest.fixture
def store():
    os.environ.setdefault("NEO4J_AUTH", "neo4j/test_password")
    try:
        kg = KnowledgeGraphStore()
        kg.get_node("__connectivity_check__")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"graph-db not reachable: {exc}")
    yield kg
    kg.close()


def test_write_and_read_round_trip(store: KnowledgeGraphStore):
    item_id = f"test-{uuid.uuid4()}"
    brand = f"brand-{uuid.uuid4()}"

    store.write_items([{"item_id": item_id, "title": "t", "price": 1.0, "main_category": "m"}])
    written = store.write_brands([(item_id, brand)])
    assert written == 1

    node = store.get_node(item_id)
    assert node is not None
    assert node.label == "Item"

    neighbors = store.neighbors(item_id, "HAS_BRAND")
    assert [n.id for n in neighbors] == [brand]

    assert store.items_by_attribute(brand=brand) == [item_id]


def test_writes_are_idempotent(store: KnowledgeGraphStore):
    item_id = f"test-{uuid.uuid4()}"
    row = {"item_id": item_id, "title": "t", "price": 1.0, "main_category": "m"}

    store.write_items([row])
    store.write_items([row])  # re-run must not duplicate the node

    neighbors_seen = store.neighbors(item_id)
    assert len(neighbors_seen) == 0  # no edges yet, but no error on rewrite either
