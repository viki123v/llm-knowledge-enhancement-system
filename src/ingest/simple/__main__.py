from __future__ import annotations

from ctypes import cast
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import argparse
from dataclasses import dataclass

from ingest.raw_data_schema import ProductMetadata, RawReview

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
PROCESSED_DIR = PROCESSED_ROOT / "simple"
PARAMS_FILE = REPO_ROOT / "params.yaml"
RUNS_FILE = PROCESSED_ROOT / "runs.json"
PIPELINE_NAME = "simple"

logger = logging.getLogger(__name__)



def write_run_params(args: IngestParams) -> None:
    """Record preprocessing pipeline name and params in runs.json."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "preprocessing_pipeline": PIPELINE_NAME,
        **args.__dict__,
    }
    with open(RUNS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info("Wrote run params to %s", RUNS_FILE)


def embeddings_dir_for_model(embedding_model: str) -> Path:
    slug = embedding_model.replace("/", "-")
    return PROCESSED_DIR / "embeddings" / slug


def load_selected_item_ids(item_split_factor: int) -> set[str]:
    """Load item ids, retaining every item_split_factor-th item."""
    logger.info(
        "Selecting item ids from %s (item_split_factor=%s)",
        ProductMetadata.__file__,
        item_split_factor,
    )
    item_ids: list[str] = []
    with open(REPO_ROOT / ProductMetadata.__file__) as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            item_ids.append(row["parent_asin"])
            if line_no % 100_000 == 0:
                logger.info("Scanned %s metadata rows for item selection", line_no)

    selected = set(item_ids[::item_split_factor])
    logger.info(
        "Selected %s / %s items (1/%s)",
        len(selected),
        len(item_ids),
        item_split_factor,
    )
    return selected


def create_item_description(selected_item_ids: set[str]) -> list[dict]:
    logger.info(
        "Building item descriptions from %s for %s selected items",
        ProductMetadata.__file__,
        len(selected_item_ids),
    )
    items = []
    with open(REPO_ROOT / ProductMetadata.__file__) as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            item_id = row["parent_asin"]
            if item_id not in selected_item_ids:
                continue
            items.append(
                {
                    "id": item_id,
                    "description": row["description"],
                }
            )
            if line_no % 100_000 == 0:
                logger.info(
                    "Scanned %s metadata rows; kept %s descriptions so far",
                    line_no,
                    len(items),
                )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "item_description.json"
    with open(out_path, "w") as f:
        json.dump(items, f)
    logger.info("Wrote %s item descriptions to %s", len(items), out_path)
    return items


def _description_to_text(description: list[str] | str | None) -> str:
    if description is None:
        return ""
    if isinstance(description, str):
        return description
    return " ".join(part for part in description if part)


def _chunked(items: list, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield start, items[start : start + chunk_size]


def _split_chunks_among_threads(
    chunks: list[tuple[int, list]], num_threads: int
) -> list[list[tuple[int, int, list]]]:
    """Partition chunks into contiguous groups, one per worker thread."""
    if not chunks:
        return []

    num_groups = min(num_threads, len(chunks))
    base, rem = divmod(len(chunks), num_groups)
    groups: list[list[tuple[int, int, list]]] = []
    offset = 0
    for group_idx in range(num_groups):
        size = base + (1 if group_idx < rem else 0)
        group_chunks = [
            (batch_idx, start, chunk_texts)
            for batch_idx, (start, chunk_texts) in enumerate(
                chunks[offset : offset + size], start=offset + 1
            )
        ]
        groups.append(group_chunks)
        offset += size
    return groups


def create_item_description_embeddings(
    items: list[dict],
    embedding_model: str,
    *,
    num_threads: int = 4,
    chunk_size: int = 64,
) -> None:
    """Embed item descriptions and write a Meta FAISS IndexFlatIP."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    item_ids = [item["id"] for item in items]
    texts = [_description_to_text(item.get("description")) for item in items]
    chunks = list(_chunked(texts, chunk_size))
    total_batches = len(chunks)
    chunk_groups = _split_chunks_among_threads(chunks, num_threads)
    num_groups = len(chunk_groups)
    out_dir = embeddings_dir_for_model(embedding_model)

    logger.info(
        "Embedding %s descriptions with %s "
        "(threads=%s, chunk_size=%s, total_batches=%s)",
        len(texts),
        embedding_model,
        num_threads,
        chunk_size,
        total_batches,
    )
    logger.info(
        "Identified %s chunk groups for %s threads "
        "(group sizes=%s)",
        num_groups,
        num_threads,
        [len(group) for group in chunk_groups],
    )

    model = SentenceTransformer(embedding_model)
    model.max_seq_length = 512
    encode_lock = Lock()
    embeddings = [None] * len(texts)

    def embed_group(
        group_idx: int, group_chunks: list[tuple[int, int, list]]
    ) -> tuple[int, int]:
        thread_label = f"thread-{group_idx}/{num_groups}"
        logger.info(
            "[%s] Starting group with %s chunks",
            thread_label,
            len(group_chunks),
        )
        for local_idx, (batch_idx, start, chunk_texts) in enumerate(
            group_chunks, start=1
        ):
            logger.info(
                "[%s] Batch %s/%s (global %s/%s) start=%s size=%s: "
                "embedding item-by-item",
                thread_label,
                local_idx,
                len(group_chunks),
                batch_idx,
                total_batches,
                start,
                len(chunk_texts),
            )
            for offset, text in enumerate(chunk_texts):
                idx = start + offset
                with encode_lock:
                    vector = model.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                embeddings[idx] = np.asarray(vector, dtype=np.float32)
                if (offset + 1) % 25 == 0 or offset + 1 == len(chunk_texts):
                    logger.info(
                        "[%s] Batch %s/%s (global %s/%s) progress=%s/%s",
                        thread_label,
                        local_idx,
                        len(group_chunks),
                        batch_idx,
                        total_batches,
                        offset + 1,
                        len(chunk_texts),
                    )
            logger.info(
                "[%s] Finished batch %s/%s (global %s/%s, start=%s size=%s)",
                thread_label,
                local_idx,
                len(group_chunks),
                batch_idx,
                total_batches,
                start,
                len(chunk_texts),
            )
        logger.info("[%s] Finished all %s assigned chunks", thread_label, len(group_chunks))
        return group_idx, len(group_chunks)

    logger.info(
        "Dispatching %s chunk groups across %s threads", num_groups, num_groups
    )
    with ThreadPoolExecutor(max_workers=num_groups or 1) as executor:
        futures = [
            executor.submit(embed_group, group_idx, group_chunks)
            for group_idx, group_chunks in enumerate(chunk_groups, start=1)
        ]
        for future in as_completed(futures):
            group_idx, size = future.result()
            logger.info(
                "[thread-%s/%s] Group complete (%s chunks)",
                group_idx,
                num_groups,
                size,
            )

    matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    logger.info("Building FAISS IndexFlatIP with shape %s", matrix.shape)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "items.faiss"
    ids_path = out_dir / "item_ids.json"
    faiss.write_index(index, str(index_path))
    with open(ids_path, "w") as f:
        json.dump(item_ids, f)
    logger.info("Wrote FAISS index to %s and ids to %s", index_path, ids_path)


def create_user_data(selected_item_ids: set[str]) -> None:
    logger.info(
        "Building user data from %s using %s selected items",
        RawReview.__file__,
        len(selected_item_ids),
    )
    reviews = []

    with open(REPO_ROOT / RawReview.__file__) as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if (
                row["verified_purchase"]
                and row["parent_asin"] in selected_item_ids
            ):
                reviews.append((row["user_id"], row["parent_asin"], row["timestamp"]))
            if line_no % 500_000 == 0:
                logger.info(
                    "Scanned %s review rows; kept %s so far",
                    line_no,
                    len(reviews),
                )

    logger.info("Loaded %s verified-purchase reviews on selected items", len(reviews))
    reviews.sort(key=lambda row: row[2])  # row[2] is timestamp

    user_items = defaultdict(set)
    user_reviews = defaultdict(list)
    for user_id, parent_asin, timestamp in reviews:
        user_items[user_id].add(parent_asin)
        user_reviews[user_id].append(
            {"parent_asin": parent_asin, "timestamp": timestamp}
        )

    valid_users = {user_id for user_id, items in user_items.items() if len(items) >= 3}
    logger.info("Kept %s users with at least 3 distinct items", len(valid_users))

    last_by_user: dict[str, dict] = {}
    purchase_history: dict[str, list] = {}

    for user_id in valid_users:
        all_reviews = user_reviews[user_id]
        if not all_reviews:
            continue
        # Last item is the latest purchase (assuming reviews are sorted by timestamp)
        *not_last, last = all_reviews
        last_by_user[user_id] = {
            "user_id": user_id,
            "parent_asin": last["parent_asin"],
            "timestamp": last["timestamp"],
        }
        # Store all purchases except the last
        purchase_history[user_id] = [
            {"parent_asin": entry["parent_asin"], "timestamp": entry["timestamp"]}
            for entry in not_last
        ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    user_item_path = PROCESSED_DIR / "user_item.json"
    history_path = PROCESSED_DIR / "user_purchase_history.json"
    with open(user_item_path, "w") as f:
        json.dump(list(last_by_user.values()), f)

    with open(history_path, "w") as f:
        json.dump(purchase_history, f)
    logger.info(
        "Wrote %s held-out rows to %s and histories to %s",
        len(last_by_user),
        user_item_path,
        history_path,
    )

@dataclass
class IngestParams:
    item_split_factor: int = None
    embedding_model: str = None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting simple ingest")

    parser = argparse.ArgumentParser(description="Simple Ingest Pipeline")
    parser.add_argument("--item_split_factor", type=int, required=True, help="Retain every Nth item")
    parser.add_argument("--embedding_model", type=str, required=True, help="HuggingFace model for embeddings")
    args = parser.parse_args(namespace=IngestParams()) 

    selected_item_ids = load_selected_item_ids(args.item_split_factor)

    logger.info("Step 1/3: create_user_data")
    create_user_data(selected_item_ids)

    logger.info("Step 2/3: create_item_description")
    items = create_item_description(selected_item_ids)

    logger.info("Step 3/3: create_item_description_embeddings")
    create_item_description_embeddings(items, args.embedding_model)

    write_run_params(args)
    logger.info("Simple ingest finished")


if __name__ == "__main__":
    main()
