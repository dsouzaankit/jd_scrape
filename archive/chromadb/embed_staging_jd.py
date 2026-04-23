#!/usr/bin/env python3
r"""
[Archived] Ingest LinkedIn job JSON into ChromaDB collection `staging_jd`.

1. Chunk each job's JSON `description` field on multi-newline boundaries (blank-line separated blocks);
   stored documents contain only that description text (no title/company prefix).
2. Embed chunks with sentence-transformers (normalized = cosine similarity via dot product).
3. Extract a synthetic "requirements" document from sections whose headers match role-related
   keywords (experience, responsibilities, qualifications, etc.).
4. Embed that requirements text per job and store per-chunk cosine similarity in metadata as
   `similarity_to_requirements` (0.0 when no requirements text was extracted).

Active vector store for this repo is DuckDB: ``script/embed_staging_jd_duckdb.py``.
Install Chroma in the venv before running this script:
  pip install -r archive/chromadb/requirements-chroma.txt

Run inside venv:
  cd <path-to-jd_scrape>
  .\\.venv\\Scripts\\Activate.ps1
  cd Z:\STUDY\web_scrape\jd_scrape\archive\chromadb
  python embed_staging_jd.py --reset --min-merge-chars 10

Defaults:
  Input : ../../data/linkedin_data_engineer_edison_50mi.json
  Store : ./chroma_persist (this archive folder)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

_ARCHIVE_DIR = Path(__file__).resolve().parent
_ROOT = _ARCHIVE_DIR.parents[1]
_SCRIPT_DIR = _ROOT / "script"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from staging_jd_core import chunk_document_for_embedding

DEFAULT_JSON = _ROOT / "data" / "linkedin_data_engineer_edison_50mi.json"
DEFAULT_CHROMA = _ARCHIVE_DIR / "chroma_persist"
COLLECTION = "staging_jd"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class STEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        arr = self._model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return arr.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--persist", type=Path, default=DEFAULT_CHROMA)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-merge-chars", type=int, default=120)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--n-results", type=int, default=5)
    args = ap.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = payload.get("jobs") or []

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
        jid = str(job["job_id"])
        requirements_by_job[jid] = req_text

    if not all_ids:
        raise SystemExit("No chunks produced from input JSON.")

    # Encode chunks
    chunk_embs = embedder.encode(
        all_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Encode requirements (one vector per job, reused for all chunks of that job)
    job_ids_unique = list(requirements_by_job.keys())
    req_texts = [requirements_by_job[j] or "(no structured requirements sections detected)" for j in job_ids_unique]
    req_embs = embedder.encode(
        req_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    req_map = {j: req_embs[i] for i, j in enumerate(job_ids_unique)}

    # Enrich metadata with cosine similarity to requirements text for the same job
    sims: list[float] = []
    for i, meta in enumerate(all_metas):
        jid = meta["job_id"]
        vec_c = chunk_embs[i].astype(np.float32)
        vec_r = req_map[jid].astype(np.float32)
        if not requirements_by_job.get(jid, "").strip():
            sims.append(0.0)
        else:
            sims.append(float(np.dot(vec_c, vec_r)))

    for meta, sim in zip(all_metas, sims):
        meta["similarity_to_requirements"] = float(sim)
        meta["has_requirements_sections"] = bool(requirements_by_job.get(meta["job_id"], "").strip())

    args.persist.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.persist))

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    st_ef = STEmbedding(args.model)
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=st_ef,
        metadata={
            "embedding_model": args.model,
            "source_json": str(args.input.resolve()),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    embeddings_list: list[list[float]] = chunk_embs.astype(float).tolist()

    batch = 100
    for start in range(0, len(all_ids), batch):
        end = start + batch
        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=all_metas[start:end],
        )

    manifest = {
        "chroma_path": str(args.persist.resolve()),
        "collection": args.collection,
        "embedding_model": args.model,
        "num_jobs": len(jobs),
        "num_vectors": len(all_ids),
        "source_json": str(args.input.resolve()),
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
        ],
    }
    manifest_path = _ARCHIVE_DIR / "staging_jd_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Collection: {args.collection}")
    print(f"Chunks (vectors): {len(all_ids)} from {len(jobs)} jobs")
    print(f"Chroma persist: {args.persist.resolve()}")
    print(f"Manifest: {manifest_path}")

    if args.query:
        q = re.sub(r"\s+", " ", args.query.strip())
        res = collection.query(
            query_texts=[q],
            n_results=min(args.n_results, len(all_ids)),
            include=["documents", "metadatas", "distances"],
        )
        print("\nQuery:", q)
        for i, doc_id in enumerate(res["ids"][0]):
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i]
            sim_rq = meta.get("similarity_to_requirements")
            print(
                f"  [{i + 1}] {meta.get('title')} @ {meta.get('company')} | "
                f"dist={dist:.4f} sim_to_req={sim_rq}"
            )


if __name__ == "__main__":
    main()
