from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase


# ponytail: nodes are matched by whichever id property they actually have
# (item_id / name / user_id). One generic query per node type avoids branching
# on node label everywhere a node id crosses a boundary; add a real index /
# per-label lookup if this ever shows up in a profiler.
def _id_match(var: str, param: str) -> str:
    return f"({var}.item_id = ${param} OR {var}.name = ${param} OR {var}.user_id = ${param})"


@dataclass(frozen=True, slots=True)
class KGNode:
    id: str
    label: str
    properties: dict[str, Any]


def _default_uri() -> str:
    return os.getenv("NEO4J_URI", "bolt://localhost:7687")


def _default_auth() -> tuple[str, str]:
    user, password = os.environ["NEO4J_AUTH"].split("/", 1)
    return user, password


def _node_id(properties: dict[str, Any]) -> str:
    return properties.get("item_id") or properties.get("name") or properties["user_id"]


class KnowledgeGraphStore:
    """Neo4j-backed graph of items, categories, brands, stores, and users."""

    def __init__(self, uri: str | None = None, auth: tuple[str, str] | None = None):
        self._driver = GraphDatabase.driver(uri or _default_uri(), auth=auth or _default_auth())

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> KnowledgeGraphStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def write_items(self, items: list[dict]) -> int:
        return self._write_count(
            """
            UNWIND $items AS item
            MERGE (i:Item {item_id: item.item_id})
            SET i.title = item.title, i.price = item.price, i.main_category = item.main_category
            RETURN count(*) AS n
            """,
            items=items,
        )

    def write_in_category(self, edges: list[tuple[str, str]]) -> int:
        return self._write_count(
            """
            UNWIND $edges AS edge
            MATCH (i:Item {item_id: edge[0]})
            MERGE (c:Category {name: edge[1]})
            MERGE (i)-[:IN_CATEGORY]->(c)
            RETURN count(*) AS n
            """,
            edges=[list(e) for e in edges],
        )

    def write_subcategory_of(self, edges: list[tuple[str, str]]) -> int:
        return self._write_count(
            """
            UNWIND $edges AS edge
            MERGE (child:Category {name: edge[0]})
            MERGE (parent:Category {name: edge[1]})
            MERGE (child)-[:SUBCATEGORY_OF]->(parent)
            RETURN count(*) AS n
            """,
            edges=[list(e) for e in edges],
        )

    def write_brands(self, edges: list[tuple[str, str]]) -> int:
        return self._write_count(
            """
            UNWIND $edges AS edge
            MATCH (i:Item {item_id: edge[0]})
            MERGE (b:Brand {name: edge[1]})
            MERGE (i)-[:HAS_BRAND]->(b)
            RETURN count(*) AS n
            """,
            edges=[list(e) for e in edges],
        )

    def write_stores(self, edges: list[tuple[str, str]]) -> int:
        return self._write_count(
            """
            UNWIND $edges AS edge
            MATCH (i:Item {item_id: edge[0]})
            MERGE (s:Store {name: edge[1]})
            MERGE (i)-[:SOLD_BY]->(s)
            RETURN count(*) AS n
            """,
            edges=[list(e) for e in edges],
        )

    def write_purchases(self, edges: list[tuple[str, str, int]]) -> int:
        return self._write_count(
            """
            UNWIND $edges AS edge
            MATCH (i:Item {item_id: edge[1]})
            MERGE (u:User {user_id: edge[0]})
            MERGE (u)-[p:PURCHASED]->(i)
            SET p.timestamp = edge[2]
            RETURN count(*) AS n
            """,
            edges=[list(e) for e in edges],
        )

    def get_node(self, node_id: str) -> KGNode | None:
        with self._driver.session() as session:
            record = session.run(
                f"MATCH (n) WHERE {_id_match('n', 'id')} RETURN n, labels(n)[0] AS label LIMIT 1",
                id=node_id,
            ).single()
        return _to_kg_node(record) if record else None

    def neighbors(self, node_id: str, relation: str | None = None) -> list[KGNode]:
        with self._driver.session() as session:
            records = session.run(
                f"""
                MATCH (n)-[r]-(m)
                WHERE {_id_match('n', 'id')} AND ($relation IS NULL OR type(r) = $relation)
                RETURN DISTINCT m AS n, labels(m)[0] AS label
                """,
                id=node_id,
                relation=relation,
            )
            return [_to_kg_node(record) for record in records]

    def items_by_attribute(
        self,
        *,
        brand: str | None = None,
        category: str | None = None,
        store: str | None = None,
    ) -> list[str]:
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (i:Item)
                WHERE ($brand IS NULL OR (i)-[:HAS_BRAND]->(:Brand {name: $brand}))
                  AND ($category IS NULL OR (i)-[:IN_CATEGORY]->(:Category {name: $category}))
                  AND ($store IS NULL OR (i)-[:SOLD_BY]->(:Store {name: $store}))
                RETURN i.item_id AS item_id
                """,
                brand=brand,
                category=category,
                store=store,
            )
            return [record["item_id"] for record in records]

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        # ponytail: 6-hop cap keeps this cheap on a small graph; raise it (or
        # add a proper index) if the graph grows enough for this to matter.
        with self._driver.session() as session:
            record = session.run(
                f"""
                MATCH (a), (b)
                WHERE {_id_match('a', 'source')} AND {_id_match('b', 'target')}
                MATCH p = shortestPath((a)-[*..6]-(b))
                RETURN [n IN nodes(p) | coalesce(n.item_id, n.name, n.user_id)] AS path
                """,
                source=source_id,
                target=target_id,
            ).single()
        return record["path"] if record else None

    def _write_count(self, query: str, **params: object) -> int:
        with self._driver.session() as session:
            record = session.run(query, **params).single()
        return record["n"] if record else 0


def _to_kg_node(record: Any) -> KGNode:
    node = record["n"]
    properties = dict(node)
    return KGNode(id=_node_id(properties), label=record["label"], properties=properties)
