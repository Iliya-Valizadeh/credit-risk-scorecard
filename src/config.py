"""Central config: paths and database connection."""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET = "TARGET"


def db_url() -> str:
    """SQLAlchemy connection string from .env."""
    user = os.getenv("PGUSER", "postgres")
    pw = os.getenv("PGPASSWORD", "")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "credit_risk")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
