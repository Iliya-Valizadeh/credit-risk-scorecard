# 0001: Adopting the house standard

Date: 2026-09-25. Status: accepted.

## Context

This repo was built before my house standard existed. The standard now lives in the
`ds-project-standard` template (tag `v1.0`). It expects a `src/<package>/` layout,
`pyproject.toml` with `uv.lock`, a Makefile with seven targets, docs checks in CI, and
a `make demo` that needs no downloads or keys.

The repo today has a flat `src/` folder that is imported as `src`, pins in
`requirements.txt` and `requirements-dev.txt`, no Makefile, and CI that runs only ruff
and pytest. `MODEL_CARD.md` and `reports/metrics.json` already exist. The trained model
file `models/pipeline.joblib` is ignored by git. The raw data
(`data/application_train.csv`, from Kaggle's Home Credit competition) is not committed.

The rule for this work is that no number changes. Before any change, I copied
`reports/metrics.json`, `reports/gender_check.json` and a list of every number in
`README.md` and `MODEL_CARD.md` into a folder outside the repo. The last step of the
retrofit diffs against that copy.

This record makes each adoption choice once, so the later steps only carry it out.

## Decisions

### 1. How to adopt the template

Options:

- Hand-copy the files I need. This gives full control and no example code to remove.
  But the repo gets no `.copier-answers.yml`, so `copier update` can never bring in
  later fixes to `tools/`. The checks would drift from the template over time.
- Run `copier copy` into this repo. This brings the example project's code along, and
  some files clash with files I already have.

Decision: run `copier copy --trust --vcs-ref v1.0 gh:Iliya-Valizadeh/ds-project-standard .`
on the branch, with these flags:

- `--exclude` for `src/**`, `tests/**` and `reports/**`, so the template's worked
  example never enters the repo.
- `--skip` for `README.md` and `MODEL_CARD.md`. Their text is mine, and later steps
  rewrite them against the saved number list. A template copy with TODO lines would
  only get in the way.
- Every other clash (`pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`) takes
  the template's version first. Then the repo's own content goes back in by hand, in
  the same commit: the data and `*.joblib` ignore rules, and the ruff notes.

Why: the answers file is the only thing that lets `tools/` stay in sync with one source.
Taking the template's version of the shared files as the base keeps future
`copier update` diffs small. A source of `gh:` rather than a local path means the
answers file works on any machine.

`copier update` does not remember `--exclude`. Without the same flags, an update would
add the example code back. So `docs/how-to/` gets a page with the exact update command,
flags included.

### 2. Folder layout

Options:

- Keep the flat `src/`. No file moves. But the code then imports a package literally
  named `src`, which works only from the repo root with the `pythonpath = ["."]` hack.
  The template's build settings, coverage settings and Makefile all expect
  `src/<package>/`, so every one of them would need a local override. Each
  `copier update` would fight those overrides again.
- Move to `src/credit_risk_scorecard/`.

Decision: move. The move is one commit that does only two things: `git mv` of each file,
and rewriting `from src.` imports to `from credit_risk_scorecard.`. The same rewrite
covers `app/streamlit_app.py` and the import lines of `notebooks/01_eda.ipynb`. The
notebook is not re-run, so its saved outputs stay as they are. The `pythonpath` hack
goes away, because `uv sync` installs the package.

Why: an import rename cannot change a number, and the final metrics diff proves it.
The flat layout would cost a small amount forever. The move costs one reviewed commit
once.

### 3. Pinned packages

Decision:

- The runtime packages go into `[project] dependencies` with the same `==` versions as
  `requirements.txt`. Exact pins stay visible because these are the versions that
  produced the numbers.
- SQLAlchemy and psycopg2-binary become an optional extra `db`. Streamlit becomes an
  optional extra `app`. Neither is needed for the results, the tests or the demo.
- The dev tools come from the template's `dev` group, and `uv.lock` pins them.
- The old files move with `git mv` to `docs/archive/`, not deleted.

The old files pin only direct packages. A fresh lock could pick newer versions of
packages underneath them, such as scipy, and those can shift results. My local `.venv`
is the environment that made the committed numbers. So the adoption step saves its
`pip freeze` output to `docs/archive/` too, compares `uv.lock` against it, and adds a
`[tool.uv] constraint-dependencies` entry for each package whose version differs.

### 4. What `make demo` runs

The data cannot be in the repo. It comes under Kaggle's competition rules, which I have
not cleared for sharing, and the file is large. The model file cannot stand in for it either (see decision 7).

Decision: `make demo` runs `python -m credit_risk_scorecard.demo` on synthetic
applicants. The generator that `tests/conftest.py` uses today moves into the package as
`credit_risk_scorecard/synthetic.py`, and the test fixture imports it from there, so
there is one copy. The demo then calls the same functions the real run uses:
cleaning and features, the split, logistic regression, LightGBM, calibration and the
threshold. It prints a short table and scores five synthetic applicants.

Rules for the demo:

- The first line it prints says the data is synthetic and the numbers are not the
  project's results.
- It writes nothing to `reports/` or `models/`, so it can never overwrite a real
  number. If `model.run()` writes files, the demo calls the lower-level functions
  instead.
- It runs offline, without a key or any file from outside the repo. CI runs it.

### 5. What `make all` does when the data is missing

Options: skip `eval` with a warning, or fail.

Decision: fail. The `eval` target first checks for `data/application_train.csv`. If
the file is missing, it prints the path and points to `data/README.md`, then exits with
an error.

Why: `make all` is the promise that every number was just regenerated. If it skipped
`eval` and still passed, it would say "all good" without regenerating anything. CI is not
affected, because CI runs `lint`, `test`, `check-docs` and `demo` one by one, and none of
those needs the data.

`make eval` runs every script that writes a committed file under `reports/`:
`model`, `gender_check` and `logreg_check`. The adoption step checks which files each
script reads, so the order is right.

### 6. Which numbers count for "no number changed"

Decision: the final diff counts every value in `reports/metrics.json` and
`reports/gender_check.json`, except `runtime_seconds`. Run time depends on the machine,
and no claim rests on it. It also counts every number in the saved list from
`README.md` and `MODEL_CARD.md`. A number may move to another file during the rewrite,
but its value may not change.

Integers and text must match exactly. Floats should match exactly on the same machine
and the same locked versions. If a float differs only beyond the digits any document
shows, it is still written under "Numbers changed" in the status file, with its cause.
It does not pass silently.

### 7. `has_model` and `has_dataset`

Decision: `has_model: true`, `has_dataset: false`.

The repo trains two models, so it keeps `MODEL_CARD.md`. It does not build or publish a
dataset. A datasheet for Kaggle's data would have to describe how Home Credit collected
it, which I don't know, so it would be guesswork. `data/README.md` and the model card's
data section cover where the data comes from and how it is split.

`models/pipeline.joblib` stays out of git, and the model card says so plainly. The file
holds medians and most common values computed from the Kaggle rows. I have not checked
whether the competition rules allow sharing a model built from that data, so it stays
out. `make eval` rebuilds it. The Streamlit app needs that file, so the app is not the
demo.

### 8. The coverage bar

The template asks for 80% coverage of `src/`, but only for new repos. <!-- not-a-claim -->

Options:

- 80% now. <!-- not-a-claim -->
  The big functions that run the whole pipeline need the real data or heavy
  mocking. Tests written only to reach a number would say less than an honest lower
  bar.
- No bar. Coverage could then fall without anyone noticing.
- A floor that only goes up.

Decision: after the layout move, measure coverage of `credit_risk_scorecard`, round it
down to a whole percent, and set that as `--cov-fail-under`. It is never lowered. Each
new module added in this phase, such as `demo.py` and `synthetic.py`, must reach 80% on <!-- not-a-claim -->
its own in the coverage report. A `roadmap` issue tracks raising the whole package to
80%. <!-- not-a-claim -->

### 9. The eval plan, written after the results

The standard wants `docs/eval_plan.md` committed before the results. That can't happen
here: the results exist.

Decision: the file is titled "Evaluation plan (written after the results)". Its first
paragraph says it was written on 2026-09-25 from the git history, and that it is not a
plan made in advance. Each choice (the split, the metrics, the operating point, the
bootstrap) cites the commit that first fixed it. Each also says whether that commit came
before or after the first commit that reported results. Choices made after results were
seen, such as dropping `CODE_GENDER`, get their own section.

The README's "How I worked" links the file with the same label. The ML Test Score
self-score gives no credit for a plan written in advance.

### 10. Lint and type settings

Decision: use the template's ruff rules. `ruff format` runs once, in its own commit,
and that commit's hash goes into `.git-blame-ignore-revs`. Lint findings are fixed only
when the fix cannot change behavior. Any other finding gets a per-rule ignore with a
one-line reason. mypy runs on `src/` in its default mode with `check_untyped_defs`,
not in strict mode. It ignores missing type stubs for packages that ship none.
Strict mode becomes a `roadmap` issue.

Why: strict mode would mean annotating every function in code that already works. That
is a large diff with a real chance of a mistake, and it would not change any claim.

## Consequences

- `copier update` can sync `tools/` from one source, as long as it uses the flags on the
  how-to page.
- Every command in the docs changes from `python -m src.model` to
  `python -m credit_risk_scorecard.model`. The README rewrite must catch all of them.
- The lockfile may pin a transitive package newer than my local `.venv`. The freeze
  comparison in decision 3 is there to catch that before the metrics diff does.
- `make all` fails on a fresh clone until the data is downloaded. That is on purpose,
  and the error message says what to do.
- The coverage floor and the non-strict mypy setting are weaker than the template's.
  Both go into `docs/whats_weak.md` and the ML Test Score self-score.
