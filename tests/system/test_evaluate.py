from __future__ import annotations

import math

import numpy as np
import pytest

from llm_knowledge_enhancement.system import evaluate


def test_ndcg_binary_true_item_at_rank_1():
    assert evaluate._ndcg_binary(("t", "a", "b"), "t", k=3) == 1.0


def test_ndcg_binary_true_item_at_rank_2():
    assert evaluate._ndcg_binary(("a", "t", "b"), "t", k=3) == pytest.approx(
        1.0 / math.log2(3)
    )


def test_ndcg_binary_true_item_at_rank_3():
    assert evaluate._ndcg_binary(("a", "b", "t"), "t", k=3) == pytest.approx(
        1.0 / math.log2(4)
    )


def test_ndcg_binary_true_item_absent():
    assert evaluate._ndcg_binary(("a", "b", "c"), "t", k=3) == 0.0


def test_evaluate_predictions_reports_binary_ndcg_alongside_existing_fields(
    monkeypatch,
):
    catalog = {
        "t1": [1.0, 0.0],
        "a": [0.0, 1.0],
        "b": [0.0, 1.0],
        "t2": [1.0, 0.0],
        "c": [0.0, 1.0],
    }
    ids = list(catalog)

    class FakeIndex:
        item_ids = tuple(ids)
        item_id_to_row = {item_id: row for row, item_id in enumerate(ids)}

        class index:
            @staticmethod
            def reconstruct(row):
                return np.array(catalog[ids[row]], dtype=np.float32)

    monkeypatch.setattr(
        evaluate, "load_item_embedding_index", lambda embedding_model: FakeIndex()
    )

    predictions = [
        {"user_id": "u1", "predicted_item_ids": ["t1", "a", "b"]},
        {"user_id": "u2", "predicted_item_ids": ["a", "b", "t2"]},
    ]
    true_items = {"u1": "t1", "u2": "t2"}

    report = evaluate.evaluate_predictions(
        predictions, true_items, embedding_model="unused", k=3
    )

    u1, u2 = report.per_user
    assert u1.ndcg_at_3_binary == 1.0
    assert u2.ndcg_at_3_binary == pytest.approx(1.0 / math.log2(4))
    assert report.mean_ndcg_at_3_binary == pytest.approx(
        (1.0 + 1.0 / math.log2(4)) / 2
    )
    # existing fields still present and computed
    assert u1.top_one_match is True
    assert u2.top_one_match is False
