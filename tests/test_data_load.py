"""The database loader, with the database and pandas' SQL writer replaced by fakes."""

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from credit_risk_scorecard import config
from credit_risk_scorecard import model as M


@pytest.fixture
def fake_sqlalchemy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Put a stand-in `sqlalchemy` on the import path (the real one is an optional extra)."""
    urls: list[str] = []
    fake = types.ModuleType("sqlalchemy")

    def create_engine(url: str) -> str:
        urls.append(url)
        return "engine"

    fake.create_engine = create_engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake)
    return urls


def _load_module() -> Any:
    sys.modules.pop("credit_risk_scorecard.data_load", None)
    return importlib.import_module("credit_risk_scorecard.data_load")


def test_load_writes_chunks_and_skips_missing_files(
    fake_sqlalchemy: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dl = _load_module()
    pd.DataFrame({"a": range(5)}).to_csv(tmp_path / "application_train.csv", index=False)
    calls: list[tuple[str, str]] = []

    def fake_to_sql(self: pd.DataFrame, name: str, con: Any, **kw: Any) -> None:
        calls.append((name, kw["if_exists"]))

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    dl.load(chunksize=2)
    # Three chunks: the first replaces the table, the rest append.
    assert calls == [
        ("application_train", "replace"),
        ("application_train", "append"),
        ("application_train", "append"),
    ]
    out = capsys.readouterr().out
    assert "[skip] application_test.csv" in out and "[done]" in out
    assert fake_sqlalchemy and fake_sqlalchemy[0].startswith("postgresql+psycopg2://")


def test_db_url_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {
        "PGUSER": "u",
        "PGPASSWORD": "p",
        "PGHOST": "h",
        "PGPORT": "1",
        "PGDATABASE": "d",
    }.items():
        monkeypatch.setenv(k, v)
    assert config.db_url() == "postgresql+psycopg2://u:p@h:1/d"
    monkeypatch.delenv("PGUSER")
    monkeypatch.delenv("PGPORT")
    assert config.db_url().startswith("postgresql+psycopg2://postgres:p@h:5432/")


def test_load_frame_reads_the_csv_or_the_database(
    fake_sqlalchemy: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "application_train.csv", index=False)
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    assert len(M.load_frame()) == 2

    seen: list[str] = []

    def fake_read_sql(query: str, con: Any) -> pd.DataFrame:
        seen.append(query)
        return pd.DataFrame({"a": [1, 2, 3]})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    assert len(M.load_frame(from_db=True)) == 3
    assert "application_train" in seen[0] and fake_sqlalchemy
