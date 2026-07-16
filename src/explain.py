"""SHAP explainability on the boosted model (Sprint 4).

Global: summary (beeswarm) of feature impacts.
Local: a single-applicant waterfall.
Figures saved to reports/figures/.
"""
from __future__ import annotations
import numpy as np


def shap_values_for(model, X_transformed, sample=2000, seed=42):
    """Compute SHAP values on a sample of the transformed validation matrix."""
    import shap
    Xs = X_transformed[:sample]
    Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):        # some shap/LGBM versions return [class0, class1]
        sv = sv[1]
    return explainer, sv, Xs


def save_summary(sv, Xs, feature_names, path):
    import matplotlib.pyplot as plt
    import shap
    shap.summary_plot(sv, features=Xs, feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight", dpi=120); plt.close()
