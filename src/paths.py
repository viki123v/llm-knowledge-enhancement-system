from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "data" / "results"
PROCESSED_DIR = RESULTS_DIR / "processed"

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
CANONICAL_DIR = PROCESSED_DIR / "canonical"
SPLIT_DIR = PROCESSED_DIR / "split"


def run_dir(run_id: str) -> Path:
    """Directory for one experiment run: config snapshot + metrics."""
    return RESULTS_DIR / run_id


def run_config_path(run_id: str) -> Path:
    return run_dir(run_id) / "config.json"


def predictions_path(run_id: str) -> Path:
    return PREDICTIONS_DIR / run_id / "predictions.jsonl"


def predictions_manifest_path(run_id: str) -> Path:
    return PREDICTIONS_DIR / run_id / "manifest.json"


def embeddings_path(model_id: str) -> Path:
    return EMBEDDINGS_DIR / model_id / "items.npy"


def embedding_ids_path(model_id: str) -> Path:
    return EMBEDDINGS_DIR / model_id / "item_ids.json"


def metrics_output_path(run_id: str) -> Path:
    return run_dir(run_id) / "metrics.json"


def canonical_items_path() -> Path:
    return CANONICAL_DIR / "items.parquet"


def canonical_interactions_path() -> Path:
    return CANONICAL_DIR / "interactions.parquet"


def split_train_history_path(split_version: str) -> Path:
    return SPLIT_DIR / split_version / "train_history.parquet"


def split_held_out_path(split_version: str) -> Path:
    return SPLIT_DIR / split_version / "held_out.parquet"
