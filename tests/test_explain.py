"""SHAP helpers: shapes, ranking, the list-return branch and the saved figure."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from credit_risk_scorecard import explain


def test_shap_values_for_samples_rows_and_returns_matching_shapes(
    trained: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, art = trained
    explainer, sv, Xs = explain.shap_values_for(art["lgbm"], art["Xte"], sample=100, seed=1)
    assert explainer is not None
    assert Xs.shape[0] == 100
    assert sv.shape == Xs.shape


def test_shap_values_for_takes_the_positive_class_from_a_list_and_densifies_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shap

    class FakeExplainer:
        def __init__(self, model: Any) -> None:
            pass

        def shap_values(self, X: np.ndarray) -> list[np.ndarray]:
            return [np.zeros(X.shape), np.ones(X.shape)]

    monkeypatch.setattr(shap, "TreeExplainer", FakeExplainer)
    X = sp.csr_matrix(np.arange(40, dtype=float).reshape(10, 4))
    _, sv, Xs = explain.shap_values_for(object(), X, sample=5, seed=0)
    assert isinstance(Xs, np.ndarray) and Xs.shape == (5, 4)
    assert (sv == 1).all()  # the second (positive-class) array


def test_top_features_ranks_by_mean_absolute_value_and_strips_the_prefix() -> None:
    sv = np.array([[1.0, -5.0, 0.1], [-1.0, 5.0, 0.1]])
    top = explain.top_features(sv, ["num__a", "num__b", "cat__c_x"], k=2)
    assert [name for name, _ in top] == ["b", "a"]
    assert top[0][1] == pytest.approx(5.0)


def test_save_summary_writes_a_png(
    trained: tuple[dict[str, Any], dict[str, Any]], tmp_path: Path
) -> None:
    _, art = trained
    _, sv, Xs = explain.shap_values_for(art["lgbm"], art["Xte"], sample=50, seed=1)
    names = [f"num__f{i}" for i in range(Xs.shape[1])]
    out = tmp_path / "shap.png"
    explain.save_summary(sv, Xs, names, out)
    assert out.exists() and out.stat().st_size > 0
