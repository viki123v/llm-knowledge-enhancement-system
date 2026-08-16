from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import ndcg_score

from llm_knowledge_enhancement.paths import REPO_ROOT
from llm_knowledge_enhancement.system.baseline.types import UserItem
from llm_knowledge_enhancement.system.shared.item_embeddings import (
    ItemEmbeddingIndex,
    load_item_embedding_index,
    reconstruct_vector,
    reconstruct_vectors,
)

EMBEDDING_MODEL = "BAAI/bge-m3"
K = 3

DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "simple"
TRUE_ITEMS_PATH = PROCESSED_DIR / "user_item.json"
PREDICTIONS_DIR = DATA_DIR / "predictions"
EVALUATIONS_DIR = DATA_DIR / "evaluations"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UserEvaluation:
    user_id: str
    true_item_id: str
    predicted_item_ids: tuple[str, ...]
    ndcg_at_3: float
    top_one_match: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    k: int
    n_users: int
    mean_ndcg_at_3: float
    top_one_accuracy: float
    per_user: tuple[UserEvaluation, ...]


def _load_true_items(path: Path) -> dict[str, str]:
    with path.open() as f:
        rows = json.load(f)
    return {
        item.user_id: item.parent_asin
        for item in (UserItem.from_dict(row) for row in rows)
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        predictions = json.load(f)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected a list of prediction records in {path}")
    return predictions


def _prediction_path(run_id: str) -> Path:
    return PREDICTIONS_DIR / run_id / "result.json"


def _evaluation_path(run_id: str) -> Path:
    return EVALUATIONS_DIR / run_id / "evaluation.json"


def _semantic_relevances(
    item_index: ItemEmbeddingIndex,
    true_item_id: str,
    predicted_item_ids: tuple[str, ...],
) -> np.ndarray:
    true_vector = reconstruct_vector(item_index, true_item_id)
    predicted_vectors = reconstruct_vectors(item_index, predicted_item_ids)
    scores = predicted_vectors @ true_vector
    scores = np.clip(scores, 0.0, 1.0)
    for idx, item_id in enumerate(predicted_item_ids):
        if item_id == true_item_id:
            scores[idx] = 1.0
    return scores


def _ndcg_for_ranked_relevances(relevances: np.ndarray, k: int) -> float:
    if relevances.size == 0:
        return 0.0

    ranking_scores = np.arange(relevances.size, 0, -1, dtype=np.float64)
    return float(
        ndcg_score(
            y_true=np.asarray([relevances], dtype=np.float64),
            y_score=np.asarray([ranking_scores], dtype=np.float64),
            k=k,
        )
    )


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    true_items: dict[str, str],
    embedding_model: str,
    *,
    k: int = K,
) -> EvaluationReport:
    if k < 1:
        raise ValueError("k must be at least 1")

    item_index = load_item_embedding_index(embedding_model)
    per_user: list[UserEvaluation] = []
    seen_users: set[str] = set()

    for row in predictions:
        user_id = str(row["user_id"])
        if user_id in seen_users:
            raise ValueError(f"Duplicate prediction record for user_id={user_id}")
        seen_users.add(user_id)

        if user_id not in true_items:
            raise KeyError(f"Missing held-out true item for user_id={user_id}")

        predicted_item_ids = tuple(
            str(item_id) for item_id in row["predicted_item_ids"]
        )
        if len(predicted_item_ids) == 0:
            raise ValueError(f"Empty prediction list for user_id={user_id}")
        if len(set(predicted_item_ids)) != len(predicted_item_ids):
            raise ValueError(f"Duplicate predicted item for user_id={user_id}")

        predicted_at_k = predicted_item_ids[:k]
        true_item_id = true_items[user_id]
        relevances = _semantic_relevances(item_index, true_item_id, predicted_at_k)
        per_user.append(
            UserEvaluation(
                user_id=user_id,
                true_item_id=true_item_id,
                predicted_item_ids=predicted_at_k,
                ndcg_at_3=_ndcg_for_ranked_relevances(relevances, k),
                top_one_match=predicted_at_k[0] == true_item_id,
            )
        )

    if not per_user:
        raise ValueError("No prediction records to evaluate")

    return EvaluationReport(
        k=k,
        n_users=len(per_user),
        mean_ndcg_at_3=float(np.mean([row.ndcg_at_3 for row in per_user])),
        top_one_accuracy=float(np.mean([row.top_one_match for row in per_user])),
        per_user=tuple(per_user),
    )


def write_report(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(report), f, indent=2)


def run_evaluation(run_id: str) -> EvaluationReport:
    prediction_path = _prediction_path(run_id)
    evaluation_path = _evaluation_path(run_id)

    logger.info("Evaluating predictions from %s", prediction_path)
    report = evaluate_predictions(
        _load_predictions(prediction_path),
        _load_true_items(TRUE_ITEMS_PATH),
        EMBEDDING_MODEL,
        k=K,
    )
    write_report(report, evaluation_path)
    logger.info(
        "Wrote evaluation to %s: mean_ndcg_at_%s=%.4f, top_one_accuracy=%.4f",
        evaluation_path,
        K,
        report.mean_ndcg_at_3,
        report.top_one_accuracy,
    )
    return report


def main(run_id: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_evaluation(run_id)


if __name__ == "__main__":
    main()
