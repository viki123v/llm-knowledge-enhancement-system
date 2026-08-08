import argparse
from dataclasses import dataclass

from llm_knowledge_enhancement.evaluate import run_evaluation
from llm_knowledge_enhancement.system import run_system


@dataclass
class CliArgs:
    system_model: str


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="LLM Knowledge Enhancement")
    parser.add_argument(
        "--system_model", type=str, required=True, help="System model identifier"
    )
    args = parser.parse_args()
    return CliArgs(system_model=args.system_model)


def main():
    args = parse_args()

    run_system(args.system_model)
    run_evaluation()


if __name__ == "__main__":
    main()
