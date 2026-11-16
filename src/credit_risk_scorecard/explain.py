"""SHAP explainability on the boosted model.

Global: summary (beeswarm) of feature impacts, and the top features by mean |SHAP|.
Figures saved to reports/figures/.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def shap_values_for(
    model: Any, X_transformed: Any, sample: int = 2000, seed: int = 42
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Compute SHAP values on a random sample of rows of the transformed matrix."""
    import shap

    rng = np.random.default_rng(seed)
    n = X_transformed.shape[0]
    rows = rng.choice(n, size=min(sample, n), replace=False)
    Xs = X_transformed[rows]
    Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):  # some shap/LGBM versions return [class0, class1]
        sv = sv[1]
    return explainer, sv, Xs


def top_features(
    sv: np.ndarray, feature_names: Sequence[str], k: int = 10
) -> list[tuple[str, float]]:
    """Features ranked by mean absolute SHAP value, with the column-transformer prefix removed."""
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:k]
    clean = [n.split("__", 1)[-1] for n in feature_names]
    return [(clean[i], float(imp[i])) for i in order]


def save_summary(
    sv: np.ndarray, Xs: np.ndarray, feature_names: Sequence[str], path: str | Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    clean = [n.split("__", 1)[-1] for n in feature_names]
    shap.summary_plot(sv, features=Xs, feature_names=clean, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
