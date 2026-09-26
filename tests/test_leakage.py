"""The target and the row id must never reach the model."""

import re
from pathlib import Path

from credit_risk_scorecard import features as F
from credit_risk_scorecard import model as M

SQL = Path(__file__).resolve().parents[1] / "sql" / "features.sql"


def test_target_and_id_not_in_model_features(applicants):
    (X_tr, _), _, _ = M.split(applicants)
    assert "TARGET" not in X_tr.columns
    numeric, categorical = F.split_columns(X_tr)
    assert "SK_ID_CURR" not in numeric + categorical
    names = F.build_preprocessor(X_tr).fit(X_tr).get_feature_names_out()
    assert not any("TARGET" in n or "SK_ID_CURR" in n for n in names)


def test_sql_features_do_not_read_the_target():
    # Strip comments first: only SQL that would actually run matters.
    code = re.sub(r"--.*", "", SQL.read_text(encoding="utf-8"))
    assert "TARGET" not in code.upper()


def test_splits_do_not_share_applicants(applicants):
    (a, _), (b, _), (c, _) = M.split(applicants)
    ids = [set(x["SK_ID_CURR"]) for x in (a, b, c)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(applicants)


def test_gender_is_not_a_model_input_but_stays_for_the_group_check(applicants):
    (X_tr, _), _, _ = M.split(applicants)
    assert "CODE_GENDER" in X_tr.columns  # still in the data
    names = F.build_preprocessor(X_tr).fit(X_tr).get_feature_names_out()
    assert not any("CODE_GENDER" in n for n in names)  # but not in the model
    with_gender = F.build_preprocessor(X_tr, exclude=()).fit(X_tr).get_feature_names_out()
    assert any("CODE_GENDER" in n for n in with_gender)
