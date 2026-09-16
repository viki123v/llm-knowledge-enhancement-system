from __future__ import annotations

from kg import extract
from llm_knowledge_enhancement.system.baseline.types import PurchaseEvent

META = {
    "parent_asin": "B08BBQ29N5",
    "title": "Electric Guitar",
    "price": "199.99",
    "main_category": "Musical Instruments",
    "categories": ["Musical Instruments", "Guitars", "Electric Guitars"],
    "details": {"Brand": "Fender"},
    "store": "Guitar Center",
}


def test_item_row_parses_price_and_falls_back_on_missing_text():
    row = extract.item_row({"parent_asin": "x", "price": "9.5"})
    assert row == {"item_id": "x", "title": "", "price": 9.5, "main_category": ""}


def test_item_row_price_none_when_unparseable():
    assert extract.item_row({"parent_asin": "x", "price": "Ask seller"})["price"] is None


def test_category_edges_builds_item_links_and_parent_chain():
    in_category, subcategory_of = extract.category_edges(META)
    assert in_category == [
        ("B08BBQ29N5", "Musical Instruments"),
        ("B08BBQ29N5", "Guitars"),
        ("B08BBQ29N5", "Electric Guitars"),
    ]
    assert subcategory_of == [
        ("Guitars", "Musical Instruments"),
        ("Electric Guitars", "Guitars"),
    ]


def test_category_edges_empty_when_no_categories():
    assert extract.category_edges({"parent_asin": "x"}) == ([], [])


def test_brand_edge_present_and_absent():
    assert extract.brand_edge(META) == ("B08BBQ29N5", "Fender")
    assert extract.brand_edge({"parent_asin": "x", "details": {}}) is None
    assert extract.brand_edge({"parent_asin": "x"}) is None


def test_store_edge_present_and_absent():
    assert extract.store_edge(META) == ("B08BBQ29N5", "Guitar Center")
    assert extract.store_edge({"parent_asin": "x", "store": None}) is None


def test_purchase_edges_from_history_flattens_all_users():
    history = {
        "u1": [PurchaseEvent("a", 1), PurchaseEvent("b", 2)],
        "u2": [PurchaseEvent("c", 3)],
    }
    assert extract.purchase_edges_from_history(history) == [
        ("u1", "a", 1),
        ("u1", "b", 2),
        ("u2", "c", 3),
    ]


def test_purchase_edges_from_held_out():
    held_out = [{"user_id": "u1", "parent_asin": "a", "timestamp": 5}]
    assert extract.purchase_edges_from_held_out(held_out) == [("u1", "a", 5)]
