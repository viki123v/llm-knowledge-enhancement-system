from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path("../data/amazon_musical_instruments")
ASIN2CATEGORY_JSON = DATA_DIR / "asin2category.json"
ASIN2CATEGORY_PARQUET = DATA_DIR / "asin2category.parquet"
BATCH_SIZE = 100_000


def convert_asin2category_to_parquet(
    src: Path, dst: Path, batch_size: int = BATCH_SIZE
) -> Path:
    """Turn {asin: category, ...} into a 2-column parquet file (asin, category)."""
    if dst.exists():
        print(f"skip convert — already exists: {dst}")
        return dst

    schema = pa.schema([("asin", pa.string()), ("category", pa.string())])
    asins: list[str] = []
    categories: list[str] = []
    n = 0

    with src.open("rb") as f_in, pq.ParquetWriter(dst, schema) as writer:
        for asin, category in ijson.kvitems(f_in, ""):
            asins.append(asin)
            categories.append(category)
            n += 1

            if len(asins) >= batch_size:
                writer.write_table(
                    pa.Table.from_pydict(
                        {"asin": asins, "category": categories}, schema=schema
                    )
                )
                asins.clear()
                categories.clear()
                if n % 1_000_000 == 0:
                    print(f"wrote {n:,} rows...")

        if asins:
            writer.write_table(
                pa.Table.from_pydict(
                    {"asin": asins, "category": categories}, schema=schema
                )
            )

    print(f"wrote {n:,} rows → {dst}")
    return dst


convert_asin2category_to_parquet(ASIN2CATEGORY_JSON, ASIN2CATEGORY_PARQUET)
