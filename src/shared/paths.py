from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_ROOT = DATA_DIR / "processed"
PROCESSED_DIR = PROCESSED_ROOT / "simple"

PARAMS_FILE = REPO_ROOT / "params.yaml"
RUNS_FILE = PROCESSED_ROOT / "runs.json"

USER_PURCHASE_HISTORY_PATH = PROCESSED_DIR / "user_purchase_history.json"
TRUE_ITEMS_PATH = PROCESSED_DIR / "user_item.json"

PREDICTIONS_DIR = DATA_DIR / "predictions"
EVALUATIONS_DIR = DATA_DIR / "evaluations"

CONFIG_PATH = REPO_ROOT / "src" / "llm_knowledge_enhancement" / "system" / "config.json"
