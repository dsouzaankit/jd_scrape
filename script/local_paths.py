"""
Repo-local defaults for DuckDB and optional EPUB dirs (no machine-specific paths).

Override with environment variables:
  JD_STAGING_DATA_DIR — directory containing jds_books.duckdb (default: <jd_scrape>/data)
  JD_BOOKS_DIR — batch ingest directory for *.epub (default: <jd_scrape>/books)
"""

from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def data_dir() -> Path:
    return Path(os.environ.get("JD_STAGING_DATA_DIR", str(REPO_ROOT / "data")))


def books_dir() -> Path:
    return Path(os.environ.get("JD_BOOKS_DIR", str(REPO_ROOT / "books")))
