"""W1: load raw Home Credit CSVs into PostgreSQL.

Run:  python -m src.data_load
"""
import pandas as pd
from sqlalchemy import create_engine

from .config import DATA_DIR, db_url

TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
}


def load(chunksize: int = 50_000) -> None:
    engine = create_engine(db_url())
    for table, fname in TABLES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[skip] {fname} not found — see data/README.md")
            continue
        print(f"[load] {fname} -> {table}")
        first = True
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk.to_sql(
                table, engine,
                if_exists="replace" if first else "append",
                index=False,
            )
            first = False
    print("[done]")


if __name__ == "__main__":
    load()
