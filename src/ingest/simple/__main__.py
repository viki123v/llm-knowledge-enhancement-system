from __future__ import annotations

import json
from pathlib import Path

from ingest.raw_data_schema import ProductMetadata, RawReview

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def create_item_description() -> None:
    items = []
    with open(REPO_ROOT / ProductMetadata.__file__) as f:
        for line in f:
            row = json.loads(line)
            items.append(
                {
                    "id": row["parent_asin"],
                    "description": row["description"],
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "item_description.json", "w") as f:
        json.dump(items, f)


def create_user_data() -> None:
    reviews = []
    from collections import defaultdict

    with open(REPO_ROOT / RawReview.__file__) as f:
        for line in f:
            row = json.loads(line)
            if row["verified_purchase"]:
                reviews.append((row["user_id"], row["parent_asin"], row["timestamp"]))

    reviews.sort(key=lambda row: row[2])  # row[2] is timestamp

    user_items = defaultdict(set)
    user_reviews = defaultdict(list)
    for user_id, parent_asin, timestamp in reviews:
        user_items[user_id].add(parent_asin)
        user_reviews[user_id].append(
            {"parent_asin": parent_asin, "timestamp": timestamp}
        )

    valid_users = {user_id for user_id, items in user_items.items() if len(items) >= 3}

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
    with open(PROCESSED_DIR / "user_item.json", "w") as f:
        json.dump(list(last_by_user.values()), f)

    with open(PROCESSED_DIR / "user_purchase_history.json", "w") as f:
        json.dump(purchase_history, f)


def main() -> None:
    create_user_data()
    create_item_description()


if __name__ == "__main__":
    main()
