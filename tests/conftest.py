"""The tests never need the Kaggle data, so they run in CI."""

from typing import Any

import pandas as pd
import pytest

from credit_risk_scorecard import model as M
from credit_risk_scorecard.synthetic import make_applicants


@pytest.fixture
def applicants() -> pd.DataFrame:
    return make_applicants()


@pytest.fixture(scope="session")
def trained() -> tuple[dict[str, Any], dict[str, Any]]:
    """One fitted run on synthetic data (metrics, artifacts), shared by the slower tests.

    Tests must treat it as read-only.
    """
    art: dict[str, Any] = {}
    m = M.run(make_applicants(), n_boot=20, figures=False, save=False, artifacts=art)
    return m, art
