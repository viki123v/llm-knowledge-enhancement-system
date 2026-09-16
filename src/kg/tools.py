"""LLM-facing tool schemas for the knowledge graph, and their dispatch.

Direct function-calling, not MCP: this project's own chatbot code (future
PR) calls an OpenAI-compatible chat API with `tools=TOOLS`, gets a tool call
back, and routes it through `dispatch`. See docs/superpowers/specs/
2026-09-16-knowledge-graph-pipeline-design.md for why.
"""

from __future__ import annotations

from typing import Any

from kg.store import KGNode, KnowledgeGraphStore

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "kg_get_node",
            "description": (
                "Look up one node in the knowledge graph (item, brand, category, "
                "store, or user) by id and return its properties. Use for a direct "
                "fact about a single known id, not for similarity questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_neighbors",
            "description": (
                "List nodes directly connected to a given node, optionally "
                "filtered by relation (IN_CATEGORY, SUBCATEGORY_OF, HAS_BRAND, "
                "SOLD_BY, PURCHASED). Use for explicit relationship questions "
                "like 'what brand is this' or 'what did this user buy', which "
                "vector similarity cannot answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "relation": {"type": "string"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_items_by_attribute",
            "description": (
                "Find items matching a brand, category, and/or store. Use for "
                "'what else does this brand/store sell' or 'what's in this "
                "category' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string"},
                    "category": {"type": "string"},
                    "store": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_shortest_path",
            "description": (
                "Find the shortest connection path between two nodes. Use for "
                "'is there any connection between X and Y' questions, which have "
                "no vector-similarity answer since a path isn't a distance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                },
                "required": ["source_id", "target_id"],
            },
        },
    },
]


def dispatch(store: KnowledgeGraphStore, name: str, arguments: dict[str, Any]) -> Any:
    if name == "kg_get_node":
        node = store.get_node(arguments["node_id"])
        return _node_to_dict(node) if node else None
    if name == "kg_neighbors":
        nodes = store.neighbors(arguments["node_id"], arguments.get("relation"))
        return [_node_to_dict(node) for node in nodes]
    if name == "kg_items_by_attribute":
        return store.items_by_attribute(
            brand=arguments.get("brand"),
            category=arguments.get("category"),
            store=arguments.get("store"),
        )
    if name == "kg_shortest_path":
        return store.shortest_path(arguments["source_id"], arguments["target_id"])
    raise ValueError(f"Unknown tool: {name}")


def _node_to_dict(node: KGNode) -> dict:
    return {"id": node.id, "label": node.label, **node.properties}
