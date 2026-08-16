import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ingest.simple.summarizer import create_prompt
from shared.llms import DeepSeekProvider

logger = logging.getLogger(__file__)
MAX_CHARS = 512
REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "simple"


def _description_to_text(description: list[str] | str | None) -> str:
    if description is None:
        return ""
    if isinstance(description, str):
        return description
    return " ".join(part for part in description if part)


def _get_provider() -> DeepSeekProvider:
    api_key = os.getenv("API_KEY")
    if api_key is None:
        raise ValueError("env API_KEY not present")
    return DeepSeekProvider(api_key=api_key)


def _cap_summary(summary: str) -> str:
    if len(summary) <= MAX_CHARS:
        return summary
    return summary[:MAX_CHARS].rstrip()


def process_items(
    items: list[tuple[int, dict]],
    total_items: int,
    progress_lock: Lock,
    processed_count: list[int],
    provider: DeepSeekProvider | None = None,
) -> int:
    summarized = 0
    for item_idx, item in items:
        logger.info("Processing item %s/%s", item_idx, total_items)
        description = _description_to_text(item.get("description"))

        if len(description) <= MAX_CHARS:
            item["summary"] = description
            with progress_lock:
                processed_count[0] += 1
                logger.info("Processed %s/%s items", processed_count[0], total_items)
            continue

        if provider is None:
            provider = _get_provider()
        summarize_prompt = create_prompt(description, max_chars=MAX_CHARS)
        summary = provider.generate_text(summarize_prompt)
        item["summary"] = _cap_summary(summary or description)
        summarized += 1
        with progress_lock:
            processed_count[0] += 1
            logger.info("Processed %s/%s items", processed_count[0], total_items)
    return summarized


def _split_items_among_threads(
    items: list[dict], num_threads: int
) -> list[list[tuple[int, dict]]]:
    if isinstance(items, dict):
        raise TypeError(
            "_split_items_among_threads expected a list of item dictionaries, "
            "but received a single item dictionary"
        )

    if not items:
        return []

    num_groups = min(max(num_threads, 1), len(items))
    base, remainder = divmod(len(items), num_groups)
    groups = []
    start = 0
    indexed_items = list(enumerate(items, start=1))
    for group_idx in range(num_groups):
        size = base + (1 if group_idx < remainder else 0)
        groups.append(indexed_items[start : start + size])
        start += size
    return groups


def cap_descriptions(
    items: list[dict], num_threads: int = 4, write_to_file: bool = False
) -> list[dict]:
    grouped_items = _split_items_among_threads(items, num_threads)
    total_items = len(items)
    progress_lock = Lock()
    processed_count = [0]

    logger.info(
        "Processing %s item descriptions across %s thread groups",
        total_items,
        len(grouped_items),
    )

    with ThreadPoolExecutor(max_workers=len(grouped_items) or 1) as executor:
        futures = [
            executor.submit(
                process_items,
                group,
                total_items,
                progress_lock,
                processed_count,
            )
            for group in grouped_items
        ]
        for group_idx, future in enumerate(as_completed(futures), start=1):
            summarized = future.result()
            logger.info(
                "[thread-%s/%s] Group complete; summarized %s descriptions",
                group_idx,
                len(grouped_items),
                summarized,
            )

    if write_to_file:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "item_description.json"
        with open(out_path, "w") as f:
            json.dump(items, f)
        logger.info("Wrote %s capped item descriptions to %s", len(items), out_path)

    return items
