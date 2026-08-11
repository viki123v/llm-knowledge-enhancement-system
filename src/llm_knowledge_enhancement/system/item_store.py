from __future__ import annotations

import numpy as np

from llm_knowledge_enhancement.shared.item_embeddings import (
    load_item_embedding_index,
    reconstruct_vectors,
)


class ItemStore:
    """Lookup and nearest-neighbor search over item embeddings.

    Thin wrapper around the FAISS index in shared.item_embeddings; holds no
    logic of its own beyond delegating.
    """

    def __init__(self, embedding_model: str):
        self._index = load_item_embedding_index(embedding_model)

    def get_vectors(self, item_ids: tuple[str, ...]) -> np.ndarray:
        return reconstruct_vectors(self._index, item_ids)

    def search(
        self, query_vectors: np.ndarray, top_n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        search_k = min(len(self), top_n)
        return self._index.index.search(query_vectors, search_k)

    def item_id_at(self, row_id: int) -> str:
        return self._index.item_ids[row_id]

    def __len__(self) -> int:
        return self._index.index.ntotal
