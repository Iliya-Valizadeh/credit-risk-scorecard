# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and each release matches a
git tag.

## [1.0.0] - 2026-09-26

### Added

- Retrofitted the repo to the house standard from `ds-project-standard` (2026-09-25):
  moved to a `src/credit_risk_scorecard/` package layout, `pyproject.toml` with a
  `uv.lock`, a seven-target Makefile, mypy and doc checks in CI, and a `make demo` that
  needs no download or key.
- Wrote the judgment docs the standard asks for, from the git history: the evaluation
  plan (`docs/eval_plan.md`), five decision records (`docs/decisions/`), the ranked
  weaknesses list (`docs/whats_weak.md`) and the ML Test Score self-score
  (`docs/ml_test_score.md`).
- Rewrote `README.md` to the nine-section skeleton, added the Diátaxis docs
  (`docs/tutorial.md`, `docs/how-to/`, `docs/reference.md`, `docs/explanation.md`) and
  the glossary, and traced every number in the docs to a source in `CLAIMS.md`.

### Fixed

- Centred numeric columns so the logistic regression converges.
- Added Platt and isotonic [calibration](docs/glossary.md#calibration) on a held-out split, since the class-weighted
  raw scores overstated every applicant's risk.
- Moved the operating point from a fixed score cutoff to a 20% [decline rate](docs/glossary.md#decline-rate), because
  the fixed cutoff declined far more applicants once run on the full data.
- Dropped `CODE_GENDER` as a model input (2026-09-25); kept it in the data only for
  the group check.

## History before this changelog existed

These are the earlier milestones, read from the git log rather than written down at
the time:

- 2026-07-16 (`d2d9e9c`): first working pipeline, logistic regression and LightGBM on
  a data sample, with a model-risk write-up.
- 2026-09-24 (`952b03b`, `7f4f511`): first full-data run, with the train/calibration/
  test split, generated reports, and the move to a 20% decline rate.
- 2026-09-25 (`db53b26` onward): full-data calibration and [SHAP](docs/glossary.md#shap) plots, the gender and
  age group check, and dropping `CODE_GENDER` from the model.
