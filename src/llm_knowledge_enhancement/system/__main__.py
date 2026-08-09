from __future__ import annotations
import json

from llm_knowledge_enhancement.system.utils import load_user_purchase_history
from paths import REPO_ROOT
from datetime import datetime


def run_system(system_model: str, preprocessing_strategy: str):
    user_purchase_history = load_user_purchase_history()
    system = None 
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if system_model == "item_description_ranker":
        from llm_knowledge_enhancement.system.item_description_ranker import run
        system = run 

    if system is None:
        raise ValueError(f"Unknown system model: {system_model}")

    result = [] 
    for user in user_purchase_history.keys():
        result.append(
            system(user, user_purchase_history[user])
        ) 

    with open(REPO_ROOT / "data" / "processed" / "simple" / "system_result.json", "w") as f:
        json.dump(result, f)
