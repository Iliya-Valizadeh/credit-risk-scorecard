"""The tests never need the Kaggle data, so they run in CI."""

import pytest

from credit_risk_scorecard.synthetic import make_applicants


@pytest.fixture
def applicants():
    return make_applicants()
