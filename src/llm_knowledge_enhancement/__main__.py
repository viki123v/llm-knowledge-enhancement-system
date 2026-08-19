from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

from llm_knowledge_enhancement.system.baseline import run_system
from llm_knowledge_enhancement.system.evaluate import run_evaluation
from shared.paths import CONFIG_PATH

logger = logging.getLogger(__name__)


@dataclass
class CliArgs:
    system_model: str


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="LLM Knowledge Enhancement")
    parser.add_argument(
        "--system-model", type=str, required=True, help="System model identifier"
    )
    args = parser.parse_args()
    return CliArgs(system_model=args.system_model)


def load_system_config(system_model: str) -> dict:
    with CONFIG_PATH.open() as f:
        config = json.load(f)

    return config.get("systems", {}).get(system_model, {})


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    logger.info("Starting run for system_model=%s", args.system_model)
    system_config = load_system_config(args.system_model)

    run_id = run_system(args.system_model, **system_config)
    logger.info("Starting evaluation")
    run_evaluation(run_id)
    logger.info("Finished run")


if __name__ == "__main__":
    main()
