"""
Shared DuckDB file open (Windows path variants + ATTACH fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

ATTACH_ALIAS = "jds_books"


def normalize_db_path(db_arg: Path) -> Path:
    p = Path(os.path.normpath(str(Path(db_arg).expanduser())))
    if not p.is_absolute():
        p = Path(os.path.normpath(str((Path.cwd() / p).resolve(strict=False))))
    return p


def candidate_connect_strings(p: Path) -> list[str]:
    p = Path(os.path.normpath(str(p)))
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s not in seen:
            seen.add(s)
            out.append(s)

    add(p.as_posix())
    ps = str(p)
    add(ps)
    if len(ps) >= 2 and ps[1] == ":":
        body = ps[2:].lstrip("\\/").replace("/", "\\")
        add("\\\\?\\" + ps[0] + ":" + "\\" + body)
    return out


def connect_duckdb_database(
    db_path: Path, read_only: bool, sql_table_basename: str
) -> tuple[duckdb.DuckDBPyConnection, str, str, bool]:
    """
    Returns (connection, manifest_path_posix, qualified_table_sql, used_attach_fallback).
    """
    p = normalize_db_path(db_path)
    manifest_posix = p.as_posix()
    errors: list[str] = []

    for cand in candidate_connect_strings(p):
        try:
            con = duckdb.connect(cand, read_only=read_only)
            con.execute("SELECT 1")
            return con, manifest_posix, sql_table_basename, False
        except Exception as e:
            errors.append(f"  connect({cand!r}): {e}")

    safe = manifest_posix.replace("'", "''")
    ro = " (READ_ONLY)" if read_only else ""
    try:
        con = duckdb.connect(":memory:")
        con.execute(f"ATTACH '{safe}' AS {ATTACH_ALIAS}{ro};")
        con.execute("SELECT 1")
        qualified = f"{ATTACH_ALIAS}.{sql_table_basename}"
        return con, manifest_posix, qualified, True
    except Exception as e:
        errors.append(f"  ATTACH('{safe}'): {e}")
        raise SystemExit(
            "DuckDB could not open the database file. Attempts:\n"
            + "\n".join(errors)
            + "\n\nWorkarounds:\n"
            + "  * Point at an NTFS folder on a local drive, e.g.\n"
            + '      --db "C:/Users/<you>/AppData/Local/jd_scrape/jds_books.duckdb"\n'
            + "  * Or set environment variable JD_STAGING_DATA_DIR.\n"
        ) from e
