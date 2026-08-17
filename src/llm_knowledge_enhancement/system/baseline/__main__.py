from __future__ import annotations

import json
import logging
from datetime import datetime

from llm_knowledge_enhancement.paths import REPO_ROOT
from llm_knowledge_enhancement.system.baseline.utils import (
    load_user_purchase_history,
)

logger = logging.getLogger(__name__)


def run_system(system_model: str, **kwargs):
    logger.info("Loading user purchase history")
    user_purchase_history = load_user_purchase_history()
    system = None
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        "Running system_model=%s for %s users (run_id=%s)",
        system_model,
        len(user_purchase_history),
        run_id,
    )

    preprocessing_params_path = REPO_ROOT / "data" / "processed" / "runs.json"
    logger.info("Loading preprocessing params from %s", preprocessing_params_path)
    with open(preprocessing_params_path, "r") as f:
        preprocessing_params = json.load(f)

    skippable_errors: tuple[type[Exception], ...] = ()
    if system_model == "item_description_ranker":
        from llm_knowledge_enhancement.system.baseline.item_description_ranker import (
            NoEmbeddableProfileItemsError,
            run,
        )

        system = run
        skippable_errors = (NoEmbeddableProfileItemsError,)

    if system_model == "popularity":
        from llm_knowledge_enhancement.system.baseline.popularity_ranker import run

        system = run

    if system is None:
        raise ValueError(f"Unknown system model: {system_model}")

    system_params = {
        **preprocessing_params,
        **kwargs,
    }

    if (
        system_model == "item_description_ranker"
        and "embedding_model" not in system_params
    ):
        raise ValueError(
            "item_description_ranker requires an embedding_model parameter"
        )

    results = []
    skipped_users = 0
    total_users = len(user_purchase_history)
    for idx, user in enumerate(user_purchase_history.keys(), start=1):
        if idx == 1 or idx % 1_000 == 0 or idx == total_users:
            logger.info("Processing user %s/%s", idx, total_users)
        try:
            results.append(
                system(user, user_purchase_history[user], **system_params)
            )
        except skippable_errors as exc:
            skipped_users += 1
            logger.warning("Skipping user %s: %s", user, exc)

    ratio = skipped_users / total_users

    logger.info(
        "Skipped %s / %s (%.0f%%) users with no recommendable candidates",
        skipped_users,
        total_users,
        ratio
    )

    run_folder = REPO_ROOT / "data" / "predictions" / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Writing predictions to %s", run_folder)

    with open(run_folder / "result.json", "w") as f:
        json.dump(results, f)

    with open(run_folder / "config.json", "w") as f:
        json.dump({"system_name": system_model, "args": system_params}, f)
    logger.info("Finished system run %s", run_id)

    return run_id 
