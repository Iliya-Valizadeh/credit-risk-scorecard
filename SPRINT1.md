# P1 Sprint 1 — Runbook

**Gate (done when all true):** data queryable in PostgreSQL, `notebooks/01_eda.ipynb`
run with real outputs + Findings filled in, committed and pushed.

Do these in order. Where you hit something new, that's the learning — don't skip it.

## 0. Get the repo onto your machine
Unzip the project, then:
```bash
cd credit-risk-scorecard
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## 1. PostgreSQL
Pick ONE:
- **Local install (recommended):** install PostgreSQL from https://www.postgresql.org/download/windows/
  (the EDB installer). During setup, remember the `postgres` password you set.
- **Docker (one line, if you have Docker):**
  `docker run --name pg-credit -e POSTGRES_PASSWORD=changeme -p 5432:5432 -d postgres`

Create the database:
```bash
psql -U postgres -c "CREATE DATABASE credit_risk;"
```
Then:
```bash
copy .env.example .env            # Windows (macOS/Linux: cp .env.example .env)
```
Edit `.env` and set `PGPASSWORD` to your real password.

## 2. Get the data
Follow `data/README.md`. Minimum: `application_train.csv` in `data/`.
Kaggle account + accept competition rules first.

## 3. Load into Postgres
```bash
python -m src.data_load
```
Expected: `[load] application_train.csv -> application_train` then `[done]`.
Verify it landed:
```bash
psql -U postgres -d credit_risk -c "SELECT COUNT(*) FROM application_train;"
```
You should see ~307,511 rows. **That count is your first defensible number.**

## 4. Run the EDA notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```
Run every cell top to bottom. Read the output — don't just execute. For each section,
be able to answer "what did this tell me and why does it matter for modeling?"

## 5. Fill in Findings + commit
Complete the **Findings** markdown cell with your real numbers. Then:
```bash
git add -A
git commit -m "Sprint 1: data in Postgres + EDA with findings"
git push
```

## What to bring back to the next session
- The row count you loaded.
- The default rate % and imbalance ratio.
- The 2–3 strongest raw separators you saw (and their default-rate swing).
Those seed Sprint 2 (baseline model) and the README's EDA + Results sections.

## If you get stuck
Bring me the exact error text. Common ones: Postgres auth (password mismatch in `.env`),
`psycopg2` install on Windows (already using `psycopg2-binary`, so should be fine), or
Kaggle download/rules-acceptance. Don't burn an hour solo — paste the traceback.
