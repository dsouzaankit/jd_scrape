# Match job requirements to book sections

Pipeline for **staging job postings** (JSON) and **EPUB books** into a shared **DuckDB** database with **sentence-transformer** vector embeddings, plus **SQL** examples under `script/sql/`.

## Layout

| Path                                                                     | Purpose                                                                                                                                                                            |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/`                                                                  | Staging JSON inputs and `jds_books.duckdb` (default DB). Keep large/local DBs gitignored.                                                                                          |
| `books/`                                                                 | Add `.epub` files here for batch ingest; see `books/README.txt`.                                                                                                                   |
| `script/`                                                                | Main Python tools (`embed_staging_jd_duckdb.py`, `embed_staging_books.py`, helpers).                                                                                               |
| `script/sql/`                                                            | DuckDB SQL (e.g. `transformations.sql` — run with repo root as `cwd` or fix `ATTACH` paths).                                                                                       |
| `archive/chromadb/`                                                      | Legacy Chroma-based ingest; optional and usually **not** in git.                                                                                                                   |
| `archive/script/build_session_json.py`, `archive/data/descriptions_raw/` | Legacy manual flow: merge fixed job metadata with per-job `.txt` raw descriptions. **Superseded** by `script/scrape_linkedin_jobs.py`, which puts descriptions in the scrape JSON. |

## Setup

1. **Python 3.10+** recommended.  
2. Create and activate a venv (e.g. `script/setup_venv.ps1` on Windows).  
3. Install runtime deps from `script/requirements-vector.txt` (includes DuckDB, sentence-transformers, EPUB/HTML stack, and **Playwright** for LinkedIn scraping). You still need a PyTorch build suitable for your machine for `sentence-transformers`.

**Optional environment variables** (read by `script/local_paths.py`; defaults are repo `data/` and `books/`)

| Variable              | Purpose                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `JD_STAGING_DATA_DIR` | Directory that contains `jds_books.duckdb` and related staging data. |
| `JD_BOOKS_DIR`        | Directory scanned for `*.epub` batch ingest.                         |

**Windows — PowerShell (current session only)**

```powershell
$env:JD_STAGING_DATA_DIR = "D:\my-data\job_matcher"
$env:JD_BOOKS_DIR = "D:\epubs"
```

Run your Python commands from the same terminal. Paths should exist (or your workflow should create them).

**Windows — persist for your user**

```powershell
[System.Environment]::SetEnvironmentVariable("JD_STAGING_DATA_DIR", "D:\my-data\job_matcher", "User")
[System.Environment]::SetEnvironmentVariable("JD_BOOKS_DIR", "D:\epubs", "User")
```

Restart the terminal (and Cursor) so new processes see the values.

**Windows (GUI):** Run `sysdm.cpl`, open the **Advanced** tab, click **Environment Variables**, then under **User** variables click **New** for each name and directory path.

## LinkedIn Scrape Usage

`script/scrape_linkedin_jobs.py` collects LinkedIn job cards plus each job's "About the job" text and writes one JSON payload.

Install once in your venv (same file `script/setup_venv.ps1` copies to the repo root when you run it):

```bash
pip install -r script/requirements-vector.txt
python -m playwright install chromium
```

`script/requirements-vector.txt` lists the `playwright` package; **`python -m playwright install chromium`** downloads the Chromium browser binaries (run again after recreating the venv or on a new machine).

Recommended first run (headed, persistent profile so you can stay logged in):

```bash
python script/scrape_linkedin_jobs.py --headed --user-data-dir ./.pw-linkedin-profile --pause-after-load
```

Common examples:

```bash
# Default output: data/source_jd/YYYY-MM-DD/<epoch>/linkedin_jobs.json
python script/scrape_linkedin_jobs.py --out data/custom_run.json
python script/scrape_linkedin_jobs.py --keywords "Data Engineer" --location "Edison, NJ" --distance-mi 50 --days week --max-jobs 20
python script/scrape_linkedin_jobs.py --headless --user-data-dir ./.pw-linkedin-profile
```

Key flags:

- `--keywords`, `--location`, `--distance-mi`, `--days` (`day|week|month`) control the search.
- `--max-jobs` caps collected postings.
- `--out` overrides the output JSON path (default: `data/source_jd/YYYY-MM-DD/<epoch>/linkedin_jobs.json`).
- `--headed`/`--headless` controls browser visibility.
- `--user-data-dir` reuses Chromium profile/cookies to reduce login friction.
- `--pause-after-load` waits for Enter after the search page opens so you can finish sign-in before scraping (avoids the browser closing while you type when the job list is still empty).
- `--slow-mo` adds delay per browser action for debugging.

Output shape includes search metadata and `jobs[]` records with `job_id`, `title`, `company`, `location`, `url`, and extracted `description`.

## Job postings → DuckDB

From `script/` with the venv active:

```bash
python embed_staging_jd_duckdb.py --min-merge-chars 10
```

Default input: latest `data/source_jd/<YYYY-MM-DD>/<epoch>/linkedin_jobs.json` for today UTC (override with `--input` file or date folder, or `--scrape-date`).
Table: append-only `staging_jd_raw` in `data/jds_books.duckdb` (paths configurable; see local_paths / duckdb_connect). Reconciliation SQL materializes `staging_jd_curated` for downstream queries.

### Embedding request skip (incremental ingest)

Runs avoid calling the sentence-transformer encoder for chunks whose **`content_hash`** matches the **latest non-obsolete row** for the same **`chunk_id`** in `staging_jd_raw`. That lookup uses the same "latest row" ordering as reconciliation's `staging_jd_latest`: partition by `chunk_id`, order by `coalesce(source_scraped_at_utc, ingested_at_utc)` desc, then `ingested_at_utc` desc, then `run_id` desc (`rn = 1`).

- **`embedding_source`** on each appended row records what happened:
  - **`encoded`** — this row's vector came from **`embedder.encode`** in this run.
  - **`reused_latest`** — no chunk encode; the embedding (and stored requirement similarity columns taken from reuse) came from that latest prior row because the hashes matched.

- **`--re-embed-all`** disables reuse and encodes **every** chunk (e.g. after a model swap or intentional full refresh).

Each run writes **`jds_books_duckdb_manifest.json`** next to the DuckDB file with **`encoded_chunk_count`**, **`reused_chunk_count`**, and **`re_embed_all`**. The CLI summary prints the same totals.

Example: chunks skipped vs encoded on a single ingest run (`run_id` from the manifest):

```sql
SELECT embedding_source, count(*) AS n
FROM staging_jd_raw
WHERE run_id = '<manifest run_id>'
GROUP BY embedding_source;
```

## Books (EPUB) → DuckDB

```bash
python embed_staging_books.py
python embed_staging_books.py --headers-only
python embed_staging_books.py --epub "../books/Your Book.epub"
```

**EPUB scripts:** `script/staging_books_epub.py` is the **chunking library** only — it reads EPUBs (via `ebooklib` / HTML parsing), applies filters (non-chapter spine items, footnotes, code blocks, etc.), and yields `BookChunk` records (text + ISBN, titles, section indices). It does **not** talk to DuckDB or compute embeddings. `script/embed_staging_books.py` is the **ingest driver**: it imports `iter_epub_chunks` from that module, runs **sentence-transformers**, and inserts rows into the **`staging_books`** table in `jds_books.duckdb` (batch from `books/` or a single `--epub`, with `--reset` / resume behavior documented in its docstring).

Table: staging_books in the same DuckDB file. See the script’s docstring for --reset, resume, and limits.

## SQL

Run these in order (DuckDB cwd at repo root so `data/jds_books.duckdb` ATTACH paths resolve):

1. `script/sql/jd_reconciliation_build.sql`
   - Builds `staging_jd_latest` from append-only `staging_jd_raw`
   - Computes reconciliation buckets
   - Materializes `staging_jd_curated`
2. `script/sql/transformations.sql`
   - Crosswalk query only (safe for select-all/execute)
   - Reads active rows from `staging_jd_curated`


## DuckDB Table Read/Write Map

| Component | Reads from | Writes to |
| --- | --- | --- |
| `script/embed_staging_jd_duckdb.py` | JSON input (`data/source_jd/<date>/<epoch>/linkedin_jobs.json`) | `staging_jd_raw` (append-only) |
| `script/embed_staging_books.py` | EPUB chunks from `script/staging_books_epub.py` | `staging_books` |
| `script/sql/jd_reconciliation_build.sql` | `staging_jd_raw`, `staging_jd_latest` | `staging_jd_latest` (view), `staging_jd_curated` (table) |
| `script/sql/transformations.sql` | `staging_jd_curated` (active rows), `staging_books` | query result only (no table writes) |

Notes:
- `staging_jd` is legacy and no longer updated by the current JD ingest path.
- `staging_jd_latest` is a convenience latest-row view; crosswalk uses `staging_jd_curated`.

## Sample Report

![Sample report](docs/sample_report.png)
