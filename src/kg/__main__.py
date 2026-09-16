from __future__ import annotations

import json
import logging

from dotenv import load_dotenv

from ingest.raw_data_schema import PRODUCT_METADATA_FILE
from kg import extract
from kg.store import KnowledgeGraphStore
from llm_knowledge_enhancement.system.baseline.utils import load_user_purchase_history
from shared.paths import KG_MANIFEST_PATH, REPO_ROOT, TRUE_ITEMS_PATH

logger = logging.getLogger(__name__)


def _load_held_out() -> list[dict]:
    with open(TRUE_ITEMS_PATH) as f:
        return json.load(f)


def _selected_item_ids(held_out: list[dict], history: dict) -> set[str]:
    """Only build nodes for items the rest of the pipeline actually kept."""
    ids = {row["parent_asin"] for row in held_out}
    ids.update(event.parent_asin for events in history.values() for event in events)
    return ids


def build_graph() -> dict:
    logger.info("Loading purchase artifacts")
    held_out = _load_held_out()
    history = load_user_purchase_history()
    selected_item_ids = _selected_item_ids(held_out, history)
    logger.info(
        "Building KG for %s items referenced by purchase data", len(selected_item_ids)
    )

    items, in_category, subcategory_of, brands, stores = [], [], [], [], []
    with open(REPO_ROOT / PRODUCT_METADATA_FILE) as f:
        for line in f:
            meta = json.loads(line)
            if meta["parent_asin"] not in selected_item_ids:
                continue
            items.append(extract.item_row(meta))
            item_in_category, item_subcategory_of = extract.category_edges(meta)
            in_category.extend(item_in_category)
            subcategory_of.extend(item_subcategory_of)
            if item_brand := extract.brand_edge(meta):
                brands.append(item_brand)
            if item_store := extract.store_edge(meta):
                stores.append(item_store)

    purchases = extract.purchase_edges_from_held_out(
        held_out
    ) + extract.purchase_edges_from_history(history)

    logger.info(
        "Writing %s items, %s category edges, %s brand edges, %s store edges, "
        "%s purchase edges to Neo4j",
        len(items),
        len(in_category) + len(subcategory_of),
        len(brands),
        len(stores),
        len(purchases),
    )

    with KnowledgeGraphStore() as store:
        store.write_items(items)
        n_in_category = store.write_in_category(in_category)
        n_subcategory_of = store.write_subcategory_of(subcategory_of)
        n_brands = store.write_brands(brands)
        n_stores = store.write_stores(stores)
        n_purchases = store.write_purchases(purchases)

    manifest = {
        "items": {"written": len(items)},
        "in_category_edges": {"attempted": len(in_category), "written": n_in_category},
        "subcategory_of_edges": {
            "attempted": len(subcategory_of),
            "written": n_subcategory_of,
        },
        "brand_edges": {"attempted": len(brands), "written": n_brands},
        "store_edges": {"attempted": len(stores), "written": n_stores},
        "purchase_edges": {"attempted": len(purchases), "written": n_purchases},
    }
    KG_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KG_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote KG build manifest to %s", KG_MANIFEST_PATH)
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    logger.info("Starting knowledge graph build")
    build_graph()
    logger.info("Knowledge graph build finished")


if __name__ == "__main__":
    main()
