"""
Shared chunking and metadata shaping for LinkedIn JD embedding pipelines (Chroma, DuckDB, etc.).
"""

from __future__ import annotations

import re
from typing import Any

# First line of a block must match one of these (case-insensitive), allowing trailing colon.
_REQ_HEADER_CORE = (
    r"(experience|expectations|duties|responsibilities|skills|qualifications|preferred|"
    r"role\s+summary|success\s+criteria|key\s+responsibilities|required\s+qualifications|"
    r"minimum\s+qualifications|desired\s+qualifications|what\s+you['\u2019]ll\s+do|what\s+we['\u2019]re\s+looking\s+for|"
    r"you['\u2019]ll\s+come\s+with|role\s+overview|position\s+overview|about\s+the\s+role|job\s+summary|"
    r"day\s+in\s+the\s+life|a\s+day\s+in\s+the\s+life|summary|overview|"
    r"attributes\s+for\s+a\s+successful\s+candidate|required\s+skills|preferred\s+skills|"
    r"technical\s+skills|minimum\s+experience|desired|nice\s+to\s+have|must\s+have|"
    r"qualifications\s*\(required\)|qualifications\s*\(preferred\))"
)

_EXCLUDE_HEADER = re.compile(
    r"(?i)^(about\s+us|about\s+the\s+company|about\s+fora|compensation|benefits|"
    r"equal\s+opportunity|eeo|perks|salary|pay\s+range|why\s+.+|who\s+we\s+are|"
    r"additional\s+job\s+application|work\s+authorization|requirements\s+added\s+by\s+the\s+job\s+poster)$"
)

_HEADER_MATCH = re.compile(rf"(?i)^\s*{_REQ_HEADER_CORE}\s*:?\s*$")


def split_blankline_chunks(text: str, min_merge_chars: int = 120) -> list[str]:
    """Split on one or more blank lines; merge very small trailing fragments into previous neighbor."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    chunks = [p.strip() for p in parts if p.strip()]
    if not chunks:
        return []

    merged: list[str] = []
    buf = chunks[0]
    for c in chunks[1:]:
        if len(c) < min_merge_chars and not _looks_like_section_header(c):
            buf = f"{buf}\n\n{c}"
        else:
            merged.append(buf)
            buf = c
    merged.append(buf)
    return merged


def _looks_like_section_header(block: str) -> bool:
    first = block.split("\n", 1)[0].strip()
    return bool(_HEADER_MATCH.match(first)) or bool(_EXCLUDE_HEADER.match(first))


def extract_requirements_text(description: str) -> tuple[str, list[str]]:
    description = (description or "").strip()
    if not description:
        return "", []

    matched_headers: list[str] = []
    pieces: list[str] = []

    for block in split_blankline_chunks(description, min_merge_chars=80):
        lines = block.split("\n")
        head = lines[0].strip() if lines else ""
        if _EXCLUDE_HEADER.match(head):
            continue
        if not _HEADER_MATCH.match(head):
            continue
        matched_headers.append(head)
        body = "\n".join(lines[1:]).strip()
        pieces.append(f"{head}\n{body}" if body else head)

    text = "\n\n".join(pieces).strip()
    return text, matched_headers


def chunk_document_for_embedding(
    job: dict[str, Any],
    min_merge_chars: int,
) -> tuple[list[str], list[str], list[dict[str, Any]], str, list[str]]:
    """
    Returns ids, texts (description-only chunks), metadatas (without similarity yet),
    requirements_text, requirements_headers.
    """
    job_id = str(job["job_id"])
    title = job.get("title") or ""
    company = job.get("company") or ""
    location = job.get("location") or ""
    url = job.get("url") or ""
    description = job.get("description") or ""

    req_text, req_headers = extract_requirements_text(description)

    raw_chunks = split_blankline_chunks(description, min_merge_chars=min_merge_chars)
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict[str, Any]] = []

    for i, ch in enumerate(raw_chunks):
        chunk_id = f"{job_id}__chunk_{i}"
        ids.append(chunk_id)
        texts.append(ch)
        metas.append(
            {
                "job_id": job_id,
                "chunk_index": i,
                "num_chunks": len(raw_chunks),
                "title": title,
                "company": company,
                "location": location,
                "url": url,
                "requirements_headers": " | ".join(req_headers) if req_headers else "",
            }
        )

    return ids, texts, metas, req_text, req_headers
