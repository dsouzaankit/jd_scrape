#!/usr/bin/env python3
r"""
Ingest EPUBs into DuckDB table ``staging_books``: 3-sentence chunks (default) or ``--headers-only``
(headings only: ``h1`` top-level titles, no sentence chunking — faster for debugging). Resumable via chunk_id unless --no-resume.

Default books dir (batch mode): ``<jd_scrape>/books`` — override with ``JD_BOOKS_DIR``.
Default DB (same file as JD ingest; second table): ``JD_STAGING_DATA_DIR/jds_books.duckdb``
(default ``<jd_scrape>/data/jds_books.duckdb``).

Run (from ``jd_scrape/script`` after activating your venv):
  python embed_staging_books.py --headers-only
  python embed_staging_books.py --headers-only --limit-books 2 --reset
  python embed_staging_books.py
  python embed_staging_books.py --reset   # DROP staging_books
  python embed_staging_books.py --headers-only --epub "../books/Your Book.epub"
  python embed_staging_books.py --epub book.epub --no-resume

DuckDB CLI (use a path that exists on your machine; repo layout uses ``jd_scrape/data/``):
  cd <jd_scrape>
  duckdb
  ATTACH 'data/jds_books.duckdb' AS jds_books;
  USE jds_books;
Duckdb SQL query:
  select chunk_id, document, isbn, book_title, section_title, section_index, embedding from staging_books;
  DETACH jds_books;
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import duckdb
from sentence_transformers import SentenceTransformer

from duckdb_connect import connect_duckdb_database, normalize_db_path
from local_paths import books_dir, data_dir
from staging_books_epub import BookChunk, EpubLoadError, iter_epub_chunks

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DUCKDB = data_dir() / "jds_books.duckdb"
DEFAULT_BOOKS_DIR = books_dir()
TABLE_NAME = "staging_books"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 48


def _ensure_books_table(con: duckdb.DuckDBPyConnection, reset: bool, table_sql: str) -> None:
    if reset:
        con.execute(f"DROP TABLE IF EXISTS {table_sql}")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_sql} (
            chunk_id VARCHAR PRIMARY KEY,
            document VARCHAR,
            embedding DOUBLE[],
            isbn VARCHAR NOT NULL,
            book_title VARCHAR,
            chapter_number INTEGER NOT NULL,
            section_title VARCHAR,
            section_index INTEGER NOT NULL,
            chunk_index_in_book INTEGER NOT NULL,
            num_sentences INTEGER NOT NULL,
            embedding_model VARCHAR,
            epub_path VARCHAR,
            ingested_at_utc TIMESTAMPTZ
        )
        """
    )


def _existing_chunk_ids(con: duckdb.DuckDBPyConnection, table_sql: str, isbn: str) -> set[str]:
    rows = con.execute(
        f"SELECT chunk_id FROM {table_sql} WHERE isbn = ?",
        [isbn],
    ).fetchall()
    return {r[0] for r in rows}


def _insert_batch(
    con: duckdb.DuckDBPyConnection,
    table_sql: str,
    chunks: list[BookChunk],
    embeddings: list[list[float]],
    model: str,
    ingested_at: datetime,
) -> None:
    rows = []
    for ch, emb in zip(chunks, embeddings):
        rows.append(
            (
                ch.chunk_id,
                ch.document,
                emb,
                ch.isbn,
                ch.book_title,
                ch.chapter_number,
                ch.section_title,
                ch.section_index,
                ch.chunk_index_in_book,
                ch.num_sentences,
                model,
                ch.epub_path,
                ingested_at,
            )
        )
    con.executemany(
        f"""
        INSERT INTO {table_sql} VALUES (
            ?::VARCHAR, ?::VARCHAR, ?::DOUBLE[], ?::VARCHAR, ?::VARCHAR,
            ?::INTEGER, ?::VARCHAR, ?::INTEGER, ?::INTEGER, ?::INTEGER,
            ?::VARCHAR, ?::VARCHAR, ?::TIMESTAMPTZ
        )
        """,
        rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--epub",
        type=Path,
        default=None,
        metavar="PATH",
        help="Ingest only this .epub file (ignores --books-dir and --limit-books)",
    )
    ap.add_argument("--books-dir", type=Path, default=DEFAULT_BOOKS_DIR)
    ap.add_argument("--db", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reset", action="store_true", help="DROP staging_books table before ingest")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-embed all chunks for each book (delete existing rows for that ISBN first)",
    )
    ap.add_argument("--limit-books", type=int, default=0, help="0 = all EPUBs (batch mode only)")
    ap.add_argument(
        "--headers-only",
        action="store_true",
        help="Embed only h1 headings (skip h2–h6, body, and 3-sentence chunking) for debugging",
    )
    args = ap.parse_args()

    args.db = normalize_db_path(args.db)

    resolved_single_epub: Path | None = None
    if args.epub is not None:
        epub_path = args.epub.expanduser().resolve()
        if not epub_path.is_file():
            raise SystemExit(f"Not a file: {epub_path}")
        if epub_path.suffix.lower() != ".epub":
            raise SystemExit(f"Expected a .epub file: {epub_path}")
        epubs = [epub_path]
        resolved_single_epub = epub_path
        books_dir_for_manifest = epub_path.parent
    else:
        books_dir = Path(args.books_dir).expanduser()
        epubs = sorted(books_dir.glob("*.epub"))
        if args.limit_books:
            epubs = epubs[: args.limit_books]
        if not epubs:
            raise SystemExit(f"No .epub files under {books_dir}")
        books_dir_for_manifest = books_dir

    args.db.parent.mkdir(parents=True, exist_ok=True)
    con, db_path_posix, table_sql, used_attach = connect_duckdb_database(
        args.db, read_only=False, sql_table_basename=TABLE_NAME
    )
    if used_attach:
        print("Note: DuckDB opened via ATTACH (direct connect failed).")
    if args.headers_only:
        print("Mode: headers-only (h1 only; no h2–h6, no body, no 3-sentence chunking).")

    embedder = SentenceTransformer(args.model)
    ingested_at = datetime.now(timezone.utc)
    total_new = 0

    try:
        _ensure_books_table(con, args.reset, table_sql)

        for epub_path in epubs:
            print(f"\n=== {epub_path.name} ===")
            # Peek ISBN / title from first chunk iterator pass (lightweight: parse once).
            try:
                all_for_book = list(iter_epub_chunks(epub_path, headers_only=args.headers_only))
            except EpubLoadError as e:
                print(f"  Skipped — {e}")
                continue
            if not all_for_book:
                print("  (no chunks; skipped)")
                continue

            isbn = all_for_book[0].isbn
            if args.no_resume:
                con.execute(f"DELETE FROM {table_sql} WHERE isbn = ?", [isbn])
            done = _existing_chunk_ids(con, table_sql, isbn)
            pending = [c for c in all_for_book if c.chunk_id not in done]
            if not pending:
                print(f"  All {len(all_for_book)} chunks already stored (resume).")
                continue

            print(f"  New chunks: {len(pending)} / {len(all_for_book)} (ISBN {isbn})")

            for i in range(0, len(pending), BATCH_SIZE):
                batch = pending[i : i + BATCH_SIZE]
                texts = [c.document for c in batch]
                embs = embedder.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                emb_list = embs.astype(float).tolist()
                _insert_batch(con, table_sql, batch, emb_list, args.model, ingested_at)
                total_new += len(batch)
                print(f"  … inserted batch {i // BATCH_SIZE + 1} ({len(batch)} rows)")

    finally:
        con.close()

    manifest = {
        "duckdb_path": db_path_posix,
        "table": table_sql,
        "embedding_model": args.model,
        "headers_only": args.headers_only,
        "books_dir": str(books_dir_for_manifest.resolve()),
        "single_epub": str(resolved_single_epub) if resolved_single_epub is not None else None,
        "epubs_processed": len(epubs),
        "chunks_inserted_this_run": total_new,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.db.parent / "jds_books_duckdb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. New rows this run: {total_new}")
    print(f"DuckDB: {db_path_posix}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
