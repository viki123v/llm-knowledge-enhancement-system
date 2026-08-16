from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from llm_knowledge_enhancement.paths import REPO_ROOT

PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "simple"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ItemEmbeddingIndex:
    index: object
    item_ids: tuple[str, ...]
    item_id_to_row: dict[str, int]


def embedding_dir(embedding_model: str) -> Path:
    model_slug = embedding_model.replace("/", "-")
    return PROCESSED_DIR / "embeddings" / model_slug


@lru_cache(maxsize=None)
def load_item_embedding_index(embedding_model: str) -> ItemEmbeddingIndex:
    import faiss

    embeddings_path = embedding_dir(embedding_model)
    index_path = embeddings_path / "items.faiss"
    item_ids_path = embeddings_path / "item_ids.json"
    logger.info("Loading item embedding index from %s", embeddings_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS item index: {index_path}")
    if not item_ids_path.exists():
        raise FileNotFoundError(f"Missing embedding item id map: {item_ids_path}")

    index = faiss.read_index(str(index_path))
    with item_ids_path.open() as f:
        item_ids = tuple(str(item_id) for item_id in json.load(f))

    if index.ntotal != len(item_ids):
        raise ValueError(
            f"FAISS index has {index.ntotal} vectors, but {item_ids_path} "
            f"contains {len(item_ids)} item ids"
        )
    logger.info(
        "Loaded item embedding index with %s items and dimension %s",
        index.ntotal,
        index.d,
    )

    return ItemEmbeddingIndex(
        index=index,
        item_ids=item_ids,
        item_id_to_row={item_id: row for row, item_id in enumerate(item_ids)},
    )


def reconstruct_vector(item_index: ItemEmbeddingIndex, item_id: str) -> np.ndarray:
    try:
        row_id = item_index.item_id_to_row[item_id]
    except KeyError as exc:
        raise KeyError(f"Missing embedding for item id {item_id}") from exc
    return np.asarray(item_index.index.reconstruct(row_id), dtype=np.float32)


def reconstruct_vectors(
    item_index: ItemEmbeddingIndex,
    item_ids: tuple[str, ...],
) -> np.ndarray:
    missing_item_ids = [
        item_id for item_id in item_ids if item_id not in item_index.item_id_to_row
    ]
    if missing_item_ids:
        raise KeyError(
            "Missing embeddings for items: " + ", ".join(missing_item_ids[:10])
        )

    vectors = [
        item_index.index.reconstruct(item_index.item_id_to_row[item_id])
        for item_id in item_ids
    ]
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
