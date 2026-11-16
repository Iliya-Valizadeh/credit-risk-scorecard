from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from credit_risk_scorecard import evaluate as E
from credit_risk_scorecard.calibrate import CALIBRATORS
from credit_risk_scorecard.expected_loss import expected_loss_table


@pytest.fixture
def scored() -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.4, 5000)
    y = (rng.random(5000) < p).astype(int)
    return y, p


def test_threshold_metrics_move_the_right_way(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]],
) -> None:
    y, p = scored
    t = E.threshold_table(y, p, thresholds=np.linspace(0.05, 0.35, 7))
    # Raising the threshold can only flag fewer applicants and catch fewer defaulters.
    assert (np.diff(t["flag_rate"]) <= 0).all()
    assert (np.diff(t["recall"]) <= 0).all()


def test_brier_and_ece_on_toy_examples() -> None:
    y = np.array([0, 0, 1, 1])
    assert E.brier(y, np.array([0, 0, 1, 1])) == 0
    assert E.brier(y, np.array([1, 1, 0, 0])) == 1
    assert E.ece(y, np.array([0.0, 0.0, 1.0, 1.0]), n_bins=2) == 0
    assert E.ece(y, np.full(4, 0.5), n_bins=1) == 0  # right on average
    assert E.ece(y, np.full(4, 0.9), n_bins=1) == pytest.approx(0.4)


def test_calibrated_data_has_small_ece(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]],
) -> None:
    y, p = scored
    assert E.ece(y, p) < 0.03
    assert E.ece(y, np.clip(p * 3, 0, 1)) > 0.1


@pytest.mark.parametrize("name", list(CALIBRATORS))
def test_calibrators_fix_inflated_scores(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]], name: str
) -> None:
    y, p = scored
    inflated = np.clip(p * 2.5, 0, 1)  # what class weighting does
    cal = CALIBRATORS[name]().fit(inflated[:2500], y[:2500])
    fixed = cal.predict(inflated[2500:])
    assert E.ece(y[2500:], fixed) < E.ece(y[2500:], inflated[2500:])
    assert E.brier(y[2500:], fixed) < E.brier(y[2500:], inflated[2500:])


def test_bootstrap_interval_contains_point_estimate(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]],
) -> None:
    y, p = scored
    from sklearn.metrics import roc_auc_score

    noisy = p + np.random.default_rng(2).normal(0, 0.2, len(p))
    ci = E.bootstrap_ci(y, {"a": p, "b": noisy}, roc_auc_score, n_boot=200, diff=("a", "b"))
    auc = roc_auc_score(y, p)
    assert ci["a"]["lo"] < auc < ci["a"]["hi"]
    assert ci["diff"]["lo"] > 0  # a is clearly better than b


def test_expected_loss_counts_only_approved() -> None:
    y = np.array([1, 0, 1])
    ead = np.array([100.0, 100.0, 100.0])
    raw = np.array([0.1, 0.2, 0.9])  # third applicant declined
    el = expected_loss_table(y, ead, raw, raw / 2, threshold=0.5, lgd=0.5)
    assert el["approved_share"] == pytest.approx(2 / 3)
    assert el["loss_implied_by_outcomes"] == pytest.approx(50.0)
    assert el["el_raw_pd"] == pytest.approx((0.1 + 0.2) * 0.5 * 100)
    assert el["el_calibrated_pd"] == pytest.approx(el["el_raw_pd"] / 2)


def test_plot_calibration_draws_one_line_per_model_and_saves(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]], tmp_path: Path
) -> None:
    y, p = scored
    path = tmp_path / "cal.png"
    fig = E.plot_calibration(y, {"raw": p, "shrunk": p / 2}, path=path, n_bins=5)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["raw", "shrunk", "perfect"]
    assert path.exists() and path.stat().st_size > 0


def test_plot_calibration_without_a_path_returns_the_open_figure(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]],
) -> None:
    import matplotlib.pyplot as plt

    y, p = scored
    fig = E.plot_calibration(y, {"raw": p})
    assert fig.axes[0].get_xlabel().startswith("Mean predicted PD")
    plt.close(fig)


def test_bootstrap_skips_one_class_resamples() -> None:
    # One defaulter in six rows: many resamples contain none and have no AUC.
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 0, 0, 0, 0, 1])
    s = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.9])
    ci = E.bootstrap_ci(y, {"a": s}, roc_auc_score, n_boot=200)
    assert 0 <= ci["a"]["lo"] <= ci["a"]["hi"] <= 1


def test_headline_returns_both_aucs(
    scored: tuple[NDArray[np.int_], NDArray[np.float64]],
) -> None:
    y, p = scored
    h = E.headline(y, p)
    assert set(h) == {"roc_auc", "pr_auc"}
    assert 0.5 < h["roc_auc"] < 1
