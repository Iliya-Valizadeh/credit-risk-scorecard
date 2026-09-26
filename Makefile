# One command for each step. CI runs the same targets.
# Every target runs inside the uv environment, so no global installs are needed.

PKG := credit_risk_scorecard
RUN := uv run
# docs/archive holds Iliya's own early working notes, kept as a historical record
# (CLAUDE.md rule: never delete or rewrite Iliya's files). It is left out of the
# checks below, which are about the docs this repo maintains going forward.
DOCS := README.md CLAIMS.md CHANGELOG.md AI_USAGE.md $(wildcard MODEL_CARD.md DATASHEET.md) \
	docs/eval_plan.md docs/whats_weak.md docs/glossary.md docs/ml_test_score.md \
	docs/tutorial.md docs/reference.md docs/explanation.md docs/how-to docs/decisions

.PHONY: setup lint test eval demo check-docs all

setup:
	uv sync

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

test:
	$(RUN) pytest

DATA := data/application_train.csv

# The real evaluation needs the Kaggle data (ADR 0001, decision 5). Every script here
# writes a committed file under reports/: model, then gender_check, then logreg_check.
eval:
	@if [ ! -f "$(DATA)" ]; then \
		echo "Missing $(DATA). See data/README.md for how to download it."; \
		exit 1; \
	fi
	$(RUN) python -m $(PKG).model
	$(RUN) python -m $(PKG).gender_check
	$(RUN) python -m $(PKG).logreg_check

demo:
	$(RUN) python -m $(PKG).demo

check-docs:
	$(RUN) python tools/ai_signs_check.py $(DOCS)
	$(RUN) python tools/claims_check.py $(DOCS)
	$(RUN) python tools/readability_check.py --glossary docs/glossary.md $(DOCS)
	$(RUN) python tools/links_check.py $(DOCS)
	$(RUN) python tools/readme_sections.py check README.md
	$(RUN) python tools/readme_sections.py repo-map README.md --check

all: setup lint test eval demo check-docs
