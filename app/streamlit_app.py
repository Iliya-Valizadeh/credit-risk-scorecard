"""Streamlit scoring demo (Sprint 5).

Loads models/pipeline.joblib, collects a few applicant fields, overlays them on a
median template, and returns probability of default + an approve/decline decision at
the chosen threshold.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import features as F  # noqa: E402

st.set_page_config(page_title="Credit-Risk Scorecard", page_icon=":bar_chart:")
st.title("Credit-Risk Scorecard — scoring demo")

ART = Path(__file__).resolve().parents[1] / "models" / "pipeline.joblib"
if not ART.exists():
    st.warning("No trained model found. Run `python -m src.model` first to create "
               "models/pipeline.joblib.")
    st.stop()

art = joblib.load(ART)
pre, model, template, threshold = (
    art["preprocessor"], art["model"], art["template"], art["threshold"]
)

st.caption("Enter a few fields; the rest default to a typical applicant.")
c1, c2 = st.columns(2)
with c1:
    ext2 = st.slider("EXT_SOURCE_2 (external score)", 0.0, 1.0, 0.5, 0.01)
    ext3 = st.slider("EXT_SOURCE_3 (external score)", 0.0, 1.0, 0.5, 0.01)
    income = st.number_input("Annual income (AMT_INCOME_TOTAL)", 20000, 1000000, 150000, 5000)
with c2:
    credit = st.number_input("Credit amount (AMT_CREDIT)", 50000, 4000000, 600000, 10000)
    annuity = st.number_input("Annuity (AMT_ANNUITY)", 5000, 300000, 27000, 1000)
    age = st.slider("Age", 21, 70, 40)

thr = st.slider("Decision threshold (PD above this = decline)", 0.05, 0.95, float(threshold), 0.01)

if st.button("Score applicant"):
    row = dict(template)
    row.update({
        "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity, "DAYS_BIRTH": -age * 365,
    })
    X = F.clean(pd.DataFrame([row]))
    proba = float(model.predict_proba(pre.transform(X))[:, 1][0])
    st.metric("Probability of default", f"{proba*100:.1f}%")
    if proba >= thr:
        st.error(f"DECLINE — PD {proba*100:.1f}% ≥ threshold {thr*100:.0f}%")
    else:
        st.success(f"APPROVE — PD {proba*100:.1f}% < threshold {thr*100:.0f}%")
    st.caption("Demo only — not a real lending decision.")
