"""
EPUB → text chunks (≤3 sentences), with headings and chapter indices.
Filters: non-chapter spine items, code, figures, footnotes, sidebars, etc.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from ebooklib import epub

# Filename / path hints for front-matter & back-matter (not chapter body).
_SPINE_SKIP = re.compile(
    r"(?:^|[/_])(?:"
    r"cover|copyright|title\s*page|halftitle|half-title|dedication|"
    r"toc|contents|table-?of-?contents|foreword|preface|"
    r"bibliography|glossary|index|colo(?:n|phon)|"
    r"adcard|promo|advert|series|acknowledge|"
    r"00-?fm|00-?front|fm\d|ix-\d"
    r")(?:[._-]|$)",
    re.I,
)

_JUNK_BLOCK_CLASS = re.compile(
    r"footnote|endnote|fnote|noteref|sidebar|marginnote|callout|pullquote|sidenote|"
    r"codeblock|sourcecode|listing|example[-_]box|prism|programlisting|"
    r"biblio|glossary|page-?footer|credit",
    re.I,
)

# Fixed-width lookbehind for Py3.11+ engines that reject variable-width LB.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class EpubLoadError(Exception):
    """EPUB could not be opened (e.g. manifest lists files missing from the zip)."""

    def __init__(self, path: Path, message: str):
        self.path = path
        super().__init__(f"{path.name}: {message}")


@dataclass
class BookChunk:
    chunk_id: str
    document: str
    isbn: str
    book_title: str
    chapter_number: int
    section_title: str
    section_index: int
    chunk_index_in_book: int
    num_sentences: int
    epub_path: str


def isbn_key(book: epub.EpubBook, epub_path: Path) -> str:
    for tup in book.get_metadata("DC", "identifier") or []:
        val = tup[0] if isinstance(tup, tuple) else str(tup)
        digits = re.sub(r"[^0-9Xx]", "", str(val))
        if 10 <= len(digits) <= 17:
            return digits.upper() if len(digits) in (10, 13) else digits
    h = hashlib.sha256(str(epub_path.resolve()).encode("utf-8", errors="replace")).hexdigest()[:14]
    return f"noid_{h}"


def book_title_meta(book: epub.EpubBook, epub_path: Path) -> str:
    for tup in book.get_metadata("DC", "title") or []:
        v = tup[0] if isinstance(tup, tuple) else str(tup)
        if v and str(v).strip():
            return str(v).strip()
    return epub_path.stem


def _skip_spine_item(item: epub.EpubItem) -> bool:
    parts = [getattr(item, "file_name", None), getattr(item, "get_id", lambda: None)()]
    names = " ".join(str(p) for p in parts if p)
    path = names.replace("\\", "/")
    return bool(_SPINE_SKIP.search(path))


def _decompose_junk_roots(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(["script", "style", "pre", "svg", "math", "object", "iframe"])):
        tag.decompose()
    for tag in list(soup.find_all("figure")):
        tag.decompose()
    for tag in list(soup.find_all("aside")):
        tag.decompose()
    for tag in list(
        soup.find_all(
            class_=lambda c: bool(c) and _JUNK_BLOCK_CLASS.search(" ".join(c) if isinstance(c, list) else str(c))
        )
    ):
        tag.decompose()
    for tag in list(soup.find_all(id=lambda i: bool(i) and _JUNK_BLOCK_CLASS.search(str(i)))):
        tag.decompose()
    for tag in list(soup.find_all("div", class_=re.compile(r"footnotes?", re.I))):
        tag.decompose()


def _strip_inline_code_and_noise(tag) -> None:
    for c in list(tag.find_all("code")):
        c.decompose()
    for s in list(tag.find_all("sup")):
        # Often footnote markers.
        if tag.name == "p" and len(s.get_text(strip=True)) <= 3:
            s.decompose()


def _bad_ancestor(el) -> bool:
    for par in el.parents:
        if par.name in ("pre", "code", "figure", "aside", "blockquote"):
            cls = " ".join(par.get("class") or [])
            if par.name == "blockquote" and _JUNK_BLOCK_CLASS.search(cls):
                return True
            if par.name in ("pre", "code", "figure", "aside"):
                return True
    return False


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 3 and out:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return [x for x in out if x]


def chunk_three_sentences(sentences: list[str]) -> list[tuple[str, int]]:
    """Return list of (chunk_text, num_sentences)."""
    chunks: list[tuple[str, int]] = []
    i = 0
    while i < len(sentences):
        group = sentences[i : i + 3]
        chunks.append((" ".join(group), len(group)))
        i += 3
    return chunks


def _section_blocks_to_pending(
    blocks: list[str],
    *,
    section_index: int,
    section_title: str,
) -> list[tuple[str, int, int, str]]:
    """Build pending rows: (document, num_sentences, section_index, section_title)."""
    if not blocks:
        return []
    text = "\n\n".join(blocks)
    sents = split_sentences(text)
    if not sents:
        return []
    st = section_title or ""
    return [(doc, ns, section_index, st) for doc, ns in chunk_three_sentences(sents)]


def _heading_book_chunk(
    heading_text: str,
    *,
    isbn: str,
    book_title: str,
    chapter_number: int,
    section_index: int,
    epub_path: str,
    chunk_ord: list[int],
) -> BookChunk:
    """One row per heading (debug: `headers_only` uses ``h1`` only)."""
    ht = (heading_text or "").strip()
    k = chunk_ord[0]
    chunk_ord[0] += 1
    cid = f"{isbn}__ch{chapter_number:04d}__s{section_index:04d}__k{k:05d}"
    return BookChunk(
        chunk_id=cid,
        document=ht,
        isbn=isbn,
        book_title=book_title,
        chapter_number=chapter_number,
        section_title=ht,
        section_index=section_index,
        chunk_index_in_book=k,
        num_sentences=1 if ht else 0,
        epub_path=epub_path,
    )


def iter_epub_chunks(epub_path: Path, *, headers_only: bool = False) -> Iterator[BookChunk]:
    """
    Yield BookChunk for one EPUB (deterministic chunk_ids for resumable ingest).

    ``chapter_number`` is **dense**: it increases only for spine XHTML files that produce at least one
    chunk. Skipped or empty-after-filter spine items do not consume a number (avoids gaps like 3,4,25).

    ``headers_only=True``: skip body text and 3-sentence chunking; one chunk per ``h1`` only
    (``h2``–``h6`` ignored for debug speed). Do not mix resume with full ingest for the same ISBN without
    ``--no-resume`` — chunk_ids overlap the same pattern.
    """
    path_s = str(epub_path.resolve())
    try:
        book = epub.read_epub(str(epub_path))
    except KeyError as e:
        # zipfile: OPF item href not in archive (common with bad exports).
        raise EpubLoadError(
            epub_path,
            "Manifest references a file that is not in the archive (broken or inconsistent EPUB).",
        ) from e
    except zipfile.BadZipFile as e:
        raise EpubLoadError(epub_path, "Not a valid zip/EPUB archive.") from e
    isbn = isbn_key(book, epub_path)
    title = book_title_meta(book, epub_path)
    chunk_ord = [0]
    chapter_number = 0

    for _spine_id, _linear in book.spine:
        item = book.get_item_with_id(_spine_id)
        if item is None or item.media_type != "application/xhtml+xml":
            continue
        if _skip_spine_item(item):
            continue

        raw = item.get_content()
        html = raw.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        body = soup.body
        if not body:
            continue

        _decompose_junk_roots(soup)

        section_index = 0

        if headers_only:
            elements = body.find_all("h1", recursive=True)
            heading_rows: list[tuple[str, int]] = []
            for el in elements:
                if _bad_ancestor(el):
                    continue
                if el.name != "h1":
                    continue
                ht = el.get_text(" ", strip=True)
                if not ht:
                    continue
                section_index += 1
                heading_rows.append((ht, section_index))
            if not heading_rows:
                continue
            chapter_number += 1
            for ht, si in heading_rows:
                yield _heading_book_chunk(
                    ht,
                    isbn=isbn,
                    book_title=title,
                    chapter_number=chapter_number,
                    section_index=si,
                    epub_path=path_s,
                    chunk_ord=chunk_ord,
                )
            continue

        elements = body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"], recursive=True)
        section_title = ""
        para_buf: list[str] = []
        spine_pending: list[tuple[str, int, int, str]] = []

        def flush_section() -> None:
            nonlocal para_buf
            spine_pending.extend(
                _section_blocks_to_pending(
                    para_buf,
                    section_index=section_index,
                    section_title=section_title,
                )
            )
            para_buf = []

        for el in elements:
            if _bad_ancestor(el):
                continue
            if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                flush_section()
                section_index += 1
                section_title = el.get_text(" ", strip=True)
                continue
            if el.name == "p":
                _strip_inline_code_and_noise(el)
                t = el.get_text(" ", strip=True)
                if t:
                    para_buf.append(t)
                continue
            if el.name == "li":
                _strip_inline_code_and_noise(el)
                t = el.get_text(" ", strip=True)
                if t:
                    para_buf.append(t)

        flush_section()
        if not spine_pending:
            continue
        chapter_number += 1
        for document, num_sentences, sec_idx, sec_title in spine_pending:
            k = chunk_ord[0]
            chunk_ord[0] += 1
            cid = f"{isbn}__ch{chapter_number:04d}__s{sec_idx:04d}__k{k:05d}"
            yield BookChunk(
                chunk_id=cid,
                document=document,
                isbn=isbn,
                book_title=title,
                chapter_number=chapter_number,
                section_title=sec_title,
                section_index=sec_idx,
                chunk_index_in_book=k,
                num_sentences=num_sentences,
                epub_path=path_s,
            )
