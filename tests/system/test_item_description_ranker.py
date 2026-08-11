from __future__ import annotations

import numpy as np
import pytest

from llm_knowledge_enhancement.system import item_description_ranker
from llm_knowledge_enhancement.system.candidate_retriever import retrieve_candidates
from llm_knowledge_enhancement.system.profile_builder import build_profile
from llm_knowledge_enhancement.system.ranker import rank
from llm_knowledge_enhancement.system.types import PurchaseEvent


def _event(item_id: str, timestamp: int) -> PurchaseEvent:
    return PurchaseEvent(parent_asin=item_id, timestamp=timestamp)


class FakeItemStore:
    """Brute-force cosine-similarity search over a small hand-built catalog."""

    def __init__(self, vectors_by_item_id: dict[str, list[float]]):
        self._ids = list(vectors_by_item_id)
        self._matrix = np.array(
            [vectors_by_item_id[item_id] for item_id in self._ids], dtype=np.float32
        )

    def get_vectors(self, item_ids: tuple[str, ...]) -> np.ndarray:
        rows = [self._ids.index(item_id) for item_id in item_ids]
        return self._matrix[rows]

    def search(self, query_vectors: np.ndarray, top_n: int):
        scores = query_vectors @ self._matrix.T
        row_ids = np.argsort(-scores, axis=1)[:, :top_n]
        top_scores = np.take_along_axis(scores, row_ids, axis=1)
        return top_scores, row_ids

    def item_id_at(self, row_id: int) -> str:
        return self._ids[row_id]

    def __len__(self) -> int:
        return len(self._ids)


# "guitar" and "strap" point the same way; "reed" and "clarinet" point another way.
CATALOG = {
    "guitar": [1.0, 0.0],
    "strap": [0.9, 0.1],
    "capo": [0.8, 0.2],
    "reed": [0.0, 1.0],
    "clarinet": [0.1, 0.9],
}


def test_build_profile_selects_recency_window_most_recent_first():
    history = [_event("a", 1), _event("b", 3), _event("c", 2)]
    assert build_profile(history, recent_history_size=2) == ("b", "c")


def test_build_profile_raises_on_empty_history():
    with pytest.raises(ValueError):
        build_profile([], recent_history_size=1)


def test_retrieve_candidates_excludes_purchased_items():
    store = FakeItemStore(CATALOG)
    candidates = retrieve_candidates(
        store,
        profile_item_ids=("guitar",),
        purchased_item_ids={"guitar", "strap"},
        neighbors_per_history_item=3,
    )
    assert "guitar" not in candidates
    assert "strap" not in candidates
    assert "capo" in candidates


def test_retrieve_candidates_merges_by_max_score_across_profile_items():
    store = FakeItemStore(CATALOG)
    candidates = retrieve_candidates(
        store,
        profile_item_ids=("guitar", "reed"),
        purchased_item_ids=set(),
        neighbors_per_history_item=5,
    )
    # "capo" is only close to guitar; its score must be guitar's similarity,
    # not diluted by any averaging with the reed query.
    guitar_capo_score = np.array(CATALOG["guitar"]) @ np.array(CATALOG["capo"])
    assert candidates["capo"] == pytest.approx(guitar_capo_score)


def test_rank_truncates_and_breaks_ties_deterministically_by_item_id():
    scores = {"b": 0.5, "a": 0.5, "c": 0.9}
    assert rank(scores, k=2) == ["c", "a"]


def test_run_end_to_end_uses_profile_retrieval_and_ranking(monkeypatch):
    monkeypatch.setattr(
        item_description_ranker, "ItemStore", lambda embedding_model: FakeItemStore(CATALOG)
    )
    history = [_event("guitar", 1), _event("reed", 2)]

    result = item_description_ranker.run(
        user="u1",
        user_purchase_history=history,
        embedding_model="unused",
        k=2,
    )

    assert result["user_id"] == "u1"
    assert len(result["predicted_item_ids"]) == 2
    assert set(result["predicted_item_ids"]) <= set(CATALOG) - {"guitar", "reed"}
    assert set(result["prediction_scores"]) == set(result["predicted_item_ids"])
