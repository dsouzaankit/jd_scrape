#!/usr/bin/env python3
r"""
Same pipeline as the archived Chroma ingest ``archive/chromadb/embed_staging_jd.py`` (chunking,
sentence-transformers, requirements similarity), but stores rows in DuckDB.

Table: append-only raw ``staging_jd_raw`` with document text, DOUBLE[] embedding,
``embedding_source`` (``encoded`` vs ``reused_latest``), and scalar metadata columns.
Incremental runs skip the encoder when a chunk matches the latest row's ``content_hash``;
use ``--re-embed-all`` to encode every chunk.

Run from ``job_reqs_book_matcher/script`` (activate your venv first):
  python embed_staging_jd_duckdb.py --reset --min-merge-chars 10

Defaults:
  Input : latest ``linkedin_jobs.json`` under ``<repo>/data/source_jd/<YYYY-MM-DD>/<epoch>/`` for
  ``--scrape-date`` (today UTC if omitted). Override with ``--input`` (JSON file or date folder).
  DB    : ``<job_reqs_book_matcher>/data/jds_books.duckdb`` (override with ``--db`` or ``JD_STAGING_DATA_DIR``)
  If direct open fails on a mapped drive, the script retries path forms and then uses ATTACH from :memory:.
  ATTACH catalog alias for that fallback is ``jds_books`` (tables: ``jds_books.staging_jd_raw``, etc.).

DuckDB CLI examples (paths must exist on your machine):
  cd <job_reqs_book_matcher>
  duckdb
  ATTACH 'data/jds_books.duckdb' AS jds_books;
  USE jds_books;
Duckdb SQL query:
  select chunk_id, document, requirements_headers, similarity_to_requirements from staging_jd_raw;
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

from duckdb_connect import connect_duckdb_database, normalize_db_path
from local_paths import data_dir
from staging_jd_core import chunk_document_for_embedding

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SCRAPE_JSON_NAME = "linkedin_jobs.json"
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_DUCKDB = data_dir() / "jds_books.duckdb"
TABLE_NAME = "staging_jd_raw"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_content_hash(text: str, job_id: str, chunk_index: int, embedding_model: str) -> str:
    return hashlib.sha256(
        (
            text.strip()
            + "|"
            + str(job_id)
            + "|"
            + str(chunk_index)
            + "|"
            + embedding_model
        ).encode("utf-8")
    ).hexdigest()


def _parse_iso_utc(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        t = s.replace("Z", "+00:00") if s.endswith("Z") else s
        return datetime.fromisoformat(t)
    except ValueError:
        return None


def _infer_batch_epoch_from_path(json_path: Path) -> tuple[str | None, int | None]:
    """If path looks like .../source_jd/<YYYY-MM-DD>/<epoch>/linkedin_jobs.json (or legacy .../date/epoch/), return (date, epoch)."""
    try:
        resolved = json_path.resolve()
    except OSError:
        resolved = json_path
    parent = resolved.parent
    gparent = parent.parent
    epoch_s = parent.name
    date_s = gparent.name
    epoch: int | None = int(epoch_s) if epoch_s.isdigit() else None
    batch_date = date_s if DATE_DIR_RE.match(date_s) else None
    return batch_date, epoch


def resolve_latest_scrape_json(date_folder: Path) -> Path:
    """Pick the numerically greatest epoch subdir that contains SCRAPE_JSON_NAME."""
    if not date_folder.is_dir():
        raise SystemExit(f"Scrape date folder is not a directory: {date_folder}")
    best_ep = -1
    best_path: Path | None = None
    for child in date_folder.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        ep = int(child.name)
        cand = child / SCRAPE_JSON_NAME
        if cand.is_file() and ep > best_ep:
            best_ep, best_path = ep, cand
    if best_path is None:
        raise SystemExit(
            f"No {SCRAPE_JSON_NAME} found under epoch subfolders of {date_folder}. "
            f"Expected layout: <date>/<epoch>/{SCRAPE_JSON_NAME} (date folder is usually under data/source_jd/)"
        )
    return best_path


def resolve_input_json(
    input_arg: Path | None,
    scrape_date: str,
) -> Path:
    """Resolve CLI --input to a concrete JSON path."""
    if input_arg is None:
        primary = ROOT / "data" / "source_jd" / scrape_date
        fallback = data_dir() / "source_jd" / scrape_date
        legacy_a = ROOT / "data" / scrape_date
        legacy_b = data_dir() / scrape_date
        if primary.is_dir():
            return resolve_latest_scrape_json(primary)
        if fallback.is_dir():
            return resolve_latest_scrape_json(fallback)
        if legacy_a.is_dir():
            return resolve_latest_scrape_json(legacy_a)
        if legacy_b.is_dir():
            return resolve_latest_scrape_json(legacy_b)
        raise SystemExit(
            f"No scrape folder for date {scrape_date!r}. Tried:\n  {primary}\n  {fallback}\n  {legacy_a}\n  {legacy_b}\n"
            "Pass --input path/to/file.json or --input path/to/YYYY-MM-DD folder."
        )

    p = Path(input_arg).expanduser()
    try:
        p = p.resolve(strict=False)
    except OSError:
        p = Path(input_arg).expanduser()

    if p.is_file():
        return p
    if p.is_dir():
        return resolve_latest_scrape_json(p)
    raise SystemExit(f"--input must be a JSON file or a date directory: {p}")


def scrape_run_metadata(
    json_path: Path, payload: dict[str, Any], scrape_date_fallback: str
) -> tuple[str | None, int | None, datetime | None]:
    path_batch, path_epoch = _infer_batch_epoch_from_path(json_path)
    raw_epoch = payload.get("scrape_attempt_epoch")
    if raw_epoch is not None:
        epoch = int(raw_epoch)
    else:
        epoch = path_epoch
    batch_date = path_batch
    if batch_date is None and DATE_DIR_RE.match(scrape_date_fallback):
        batch_date = scrape_date_fallback
    scraped_at = _parse_iso_utc(payload.get("scraped_at_utc"))
    return batch_date, epoch, scraped_at


def _add_columns_if_missing(con: duckdb.DuckDBPyConnection, table_sql: str) -> None:
    for col, typ in (
        ("content_hash", "VARCHAR"),
        ("run_id", "VARCHAR"),
        ("last_seen_at_utc", "TIMESTAMPTZ"),
        ("is_obsolete", "BOOLEAN"),
        ("obsoleted_at_utc", "TIMESTAMPTZ"),
        ("scrape_batch_date", "VARCHAR"),
        ("scrape_attempt_epoch", "BIGINT"),
        ("source_scraped_at_utc", "TIMESTAMPTZ"),
        ("embedding_source", "VARCHAR"),
    ):
        try:
            con.execute(f"ALTER TABLE {table_sql} ADD COLUMN IF NOT EXISTS {col} {typ}")
        except Exception:
            try:
                con.execute(f"ALTER TABLE {table_sql} ADD COLUMN {col} {typ}")
            except Exception:
                pass


def _ensure_table(con: duckdb.DuckDBPyConnection, reset: bool, table_sql: str) -> None:
    if reset:
        con.execute(f"DROP TABLE IF EXISTS {table_sql}")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_sql} (
            chunk_version_id VARCHAR PRIMARY KEY,
            chunk_id VARCHAR,
            document VARCHAR,
            embedding DOUBLE[],
            job_id VARCHAR NOT NULL,
            chunk_index INTEGER NOT NULL,
            num_chunks INTEGER NOT NULL,
            title VARCHAR,
            company VARCHAR,
            location VARCHAR,
            url VARCHAR,
            requirements_headers VARCHAR,
            similarity_to_requirements DOUBLE,
            has_requirements_sections BOOLEAN,
            embedding_model VARCHAR,
            content_hash VARCHAR,
            run_id VARCHAR,
            source_json VARCHAR,
            ingested_at_utc TIMESTAMPTZ,
            last_seen_at_utc TIMESTAMPTZ,
            is_obsolete BOOLEAN,
            obsoleted_at_utc TIMESTAMPTZ,
            scrape_batch_date VARCHAR,
            scrape_attempt_epoch BIGINT,
            source_scraped_at_utc TIMESTAMPTZ,
            embedding_source VARCHAR
        )
        """
    )
    if not reset:
        _add_columns_if_missing(con, table_sql)


def _fetch_latest_reusable_by_chunk_id(
    con: duckdb.DuckDBPyConnection,
    table_sql: str,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Latest non-obsolete row per chunk_id; reuse embedding when incoming content_hash matches."""
    if not chunk_ids:
        return {}
    unique = list(dict.fromkeys(chunk_ids))
    ph = ",".join(["?"] * len(unique))
    sql = f"""
    SELECT chunk_id, content_hash, embedding, similarity_to_requirements, has_requirements_sections
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY chunk_id
                ORDER BY coalesce(source_scraped_at_utc, ingested_at_utc) DESC,
                         ingested_at_utc DESC,
                         run_id DESC
            ) AS rn
        FROM {table_sql}
        WHERE coalesce(is_obsolete, FALSE) = FALSE
    ) t
    WHERE rn = 1 AND chunk_id IN ({ph})
    """
    rows = con.execute(sql, unique).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[str(r[0])] = {
            "content_hash": r[1],
            "embedding": r[2],
            "similarity_to_requirements": float(r[3]) if r[3] is not None else 0.0,
            "has_requirements_sections": bool(r[4]) if r[4] is not None else False,
        }
    return out


def _load_build_rows(
    all_ids: list[str],
    all_texts: list[str],
    all_metas: list[dict[str, Any]],
    embeddings_list: list[list[float]],
    embedding_model: str,
    source_json: str,
    ingested_at: datetime,
    run_id: str,
    scrape_batch_date: str | None,
    scrape_attempt_epoch: int | None,
    source_scraped_at_utc: datetime | None,
    embedding_sources: list[str],
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for i, chunk_id in enumerate(all_ids):
        meta = all_metas[i]
        rows.append(
            (
                f"{chunk_id}:{run_id}",
                chunk_id,
                all_texts[i],
                embeddings_list[i],
                meta["job_id"],
                int(meta["chunk_index"]),
                int(meta["num_chunks"]),
                meta.get("title") or "",
                meta.get("company") or "",
                meta.get("location") or "",
                meta.get("url") or "",
                meta.get("requirements_headers") or "",
                float(meta["similarity_to_requirements"]),
                bool(meta["has_requirements_sections"]),
                embedding_model,
                chunk_content_hash(all_texts[i], str(meta["job_id"]), int(meta["chunk_index"]), embedding_model),
                run_id,
                source_json,
                ingested_at,
                ingested_at,
                False,
                None,
                scrape_batch_date,
                scrape_attempt_epoch,
                source_scraped_at_utc,
                embedding_sources[i],
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"JSON file, or YYYY-MM-DD folder with <epoch>/{SCRAPE_JSON_NAME}. "
        "Default: latest epoch under data/source_jd/<--scrape-date> (repo data/ then JD_STAGING_DATA_DIR).",
    )
    ap.add_argument(
        "--scrape-date",
        type=str,
        default=None,
        help="YYYY-MM-DD used when --input is omitted (default: today UTC).",
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DUCKDB, help="DuckDB file path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-merge-chars", type=int, default=120)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument(
        "--re-embed-all",
        action="store_true",
        help="Encode every chunk even if the latest staging_jd_raw row has the same content_hash (disables reuse).",
    )
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--n-results", type=int, default=5)
    args = ap.parse_args()

    scrape_date = args.scrape_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    input_json = resolve_input_json(args.input, scrape_date)
    args.db = normalize_db_path(args.db)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = payload.get("jobs") or []

    batch_fallback = scrape_date if args.input is None else ""
    scrape_batch_date, scrape_attempt_epoch, source_scraped_at_utc = scrape_run_metadata(
        input_json, payload, batch_fallback
    )

    embedder = SentenceTransformer(args.model)

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict[str, Any]] = []
    requirements_by_job: dict[str, str] = {}

    for job in jobs:
        ids, texts, metas, req_text, _req_h = chunk_document_for_embedding(
            job, min_merge_chars=args.min_merge_chars
        )
        all_ids.extend(ids)
        all_texts.extend(texts)
        all_metas.extend(metas)
        requirements_by_job[str(job["job_id"])] = req_text

    if not all_ids:
        raise SystemExit("No chunks produced from input JSON.")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    ingested_at = datetime.now(timezone.utc)
    run_id = str(int(ingested_at.timestamp() * 1000))
    source_json_s = str(input_json.resolve())

    con, db_connect, table_sql, used_attach = connect_duckdb_database(
        args.db, read_only=False, sql_table_basename=TABLE_NAME
    )
    if used_attach:
        print("Note: DuckDB opened the file via ATTACH (direct connect failed on this path).")
    try:
        _ensure_table(con, args.reset, table_sql)
        reusable: dict[str, dict[str, Any]] = {}
        if not args.re_embed_all:
            reusable = _fetch_latest_reusable_by_chunk_id(con, table_sql, all_ids)

        embedding_sources: list[str] = []
        encode_indices: list[int] = []
        for i, cid in enumerate(all_ids):
            meta = all_metas[i]
            h = chunk_content_hash(
                all_texts[i], str(meta["job_id"]), int(meta["chunk_index"]), args.model
            )
            prev = reusable.get(cid)
            if prev and prev["content_hash"] == h:
                embedding_sources.append("reused_latest")
            else:
                embedding_sources.append("encoded")
                encode_indices.append(i)

        encoded_chunk_count = embedding_sources.count("encoded")
        reused_chunk_count = embedding_sources.count("reused_latest")

        texts_to_encode = [all_texts[i] for i in encode_indices]
        chunk_embs_new = None
        if texts_to_encode:
            chunk_embs_new = embedder.encode(
                texts_to_encode,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

        embeddings_list: list[list[float]] = []
        ei = 0
        for i in range(len(all_ids)):
            if embedding_sources[i] == "encoded":
                embeddings_list.append(chunk_embs_new[ei].astype(float).tolist())
                ei += 1
            else:
                emb = reusable[all_ids[i]]["embedding"]
                if hasattr(emb, "tolist"):
                    embeddings_list.append(emb.tolist())
                else:
                    embeddings_list.append(list(emb))

        jobs_need_req: set[str] = {str(all_metas[i]["job_id"]) for i in encode_indices}

        req_map: dict[str, Any] = {}
        if jobs_need_req:
            jid_order: list[str] = []
            seenj: set[str] = set()
            for jid in jobs_need_req:
                if jid not in seenj:
                    seenj.add(jid)
                    jid_order.append(jid)
            req_texts = [
                requirements_by_job[j] or "(no structured requirements sections detected)"
                for j in jid_order
            ]
            req_embs = embedder.encode(
                req_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            req_map = {jid_order[i]: req_embs[i] for i in range(len(jid_order))}

        for i, meta in enumerate(all_metas):
            jid = str(meta["job_id"])
            if embedding_sources[i] == "reused_latest":
                r = reusable[all_ids[i]]
                meta["similarity_to_requirements"] = r["similarity_to_requirements"]
                meta["has_requirements_sections"] = r["has_requirements_sections"]
                continue
            vec_c = np.asarray(embeddings_list[i], dtype=np.float32)
            vec_r = req_map[jid].astype(np.float32)
            if not requirements_by_job.get(jid, "").strip():
                meta["similarity_to_requirements"] = 0.0
            else:
                meta["similarity_to_requirements"] = float(np.dot(vec_c, vec_r))
            meta["has_requirements_sections"] = bool(requirements_by_job.get(jid, "").strip())

        rows = _load_build_rows(
            all_ids,
            all_texts,
            all_metas,
            embeddings_list,
            args.model,
            source_json_s,
            ingested_at,
            run_id,
            scrape_batch_date,
            scrape_attempt_epoch,
            source_scraped_at_utc,
            embedding_sources,
        )

        con.executemany(
            f"""
            INSERT INTO {table_sql} (
                chunk_version_id, chunk_id, document, embedding, job_id, chunk_index, num_chunks,
                title, company, location, url, requirements_headers,
                similarity_to_requirements, has_requirements_sections, embedding_model,
                content_hash, run_id, source_json, ingested_at_utc, last_seen_at_utc,
                is_obsolete, obsoleted_at_utc, scrape_batch_date, scrape_attempt_epoch, source_scraped_at_utc,
                embedding_source
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )
    finally:
        con.close()

    manifest = {
        "duckdb_path": db_connect,
        "table": table_sql,
        "run_id": run_id,
        "duckdb_used_attach_fallback": used_attach,
        "embedding_model": args.model,
        "num_jobs": len(jobs),
        "num_vectors": len(all_ids),
        "encoded_chunk_count": encoded_chunk_count,
        "reused_chunk_count": reused_chunk_count,
        "re_embed_all": bool(args.re_embed_all),
        "source_json": source_json_s,
        "scrape_batch_date": scrape_batch_date,
        "scrape_attempt_epoch": scrape_attempt_epoch,
        "source_scraped_at_utc": source_scraped_at_utc.isoformat() if source_scraped_at_utc else None,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_fields": [
            "similarity_to_requirements",
            "has_requirements_sections",
            "requirements_headers",
            "job_id",
            "chunk_index",
            "url",
            "title",
            "company",
            "location",
            "scrape_batch_date",
            "scrape_attempt_epoch",
            "source_scraped_at_utc",
            "content_hash",
            "embedding_source",
            "run_id",
            "last_seen_at_utc",
            "is_obsolete",
            "obsoleted_at_utc",
        ],
    }
    manifest_path = args.db.parent / "jds_books_duckdb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Table (append-only raw): {table_sql}")
    print(f"Rows: {len(all_ids)} from {len(jobs)} jobs (chunks encoded={encoded_chunk_count}, reused_latest={reused_chunk_count})")
    print(f"Source JSON: {source_json_s}")
    print(f"scrape_batch_date={scrape_batch_date!r} scrape_attempt_epoch={scrape_attempt_epoch!r}")
    print(f"DuckDB: {db_connect}")
    print(f"Manifest: {manifest_path}")

    if args.query:
        con, _dbc, qtable_sql, _qa = connect_duckdb_database(
            args.db, read_only=True, sql_table_basename=TABLE_NAME
        )
        try:
            q = re.sub(r"\s+", " ", args.query.strip())
            qv = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float64)
            rel = con.execute(
                f"""
                SELECT chunk_id, document, job_id, title, company, location,
                       similarity_to_requirements, embedding
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY chunk_id
                               ORDER BY coalesce(source_scraped_at_utc, ingested_at_utc) DESC,
                                        ingested_at_utc DESC,
                                        run_id DESC
                           ) AS rn
                    FROM {qtable_sql}
                    WHERE coalesce(is_obsolete, FALSE) = FALSE
                ) t
                WHERE rn = 1
                """
            ).fetchall()
        finally:
            con.close()

        if not rel:
            print("No rows to query.")
            return

        _chunk_ids, _docs, _jids, titles, comps, locs, sims_req, embs = zip(*rel)
        mat = np.asarray([np.asarray(e, dtype=np.float64) for e in embs], dtype=np.float64)
        scores = mat @ qv
        ordering = np.argsort(-scores)[: min(args.n_results, len(scores))]
        print("\nQuery:", q)
        for rank, idx in enumerate(ordering, start=1):
            print(
                f"  [{rank}] {titles[idx]} @ {comps[idx]} | "
                f"cosine_sim={scores[idx]:.4f} sim_to_req={sims_req[idx]}"
            )


if __name__ == "__main__":
    main()
