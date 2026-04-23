#!/usr/bin/env python3
"""
LinkedIn job search scraper — logic aligned with Chrome DevTools MCP flow used in-session.

- Builds search URL (keywords, location, distance, date posted).
- Optionally opens Distance filter and sets miles via range input (LinkedIn uses 0–5 steps).
- Scrolls the overflow:auto jobs list container to collect listing cards.
- Visits each job page and extracts the \"About the job\" section from document.body.innerText.

IMPORTANT: Automated access may violate LinkedIn's Terms of Service. Prefer official APIs for
production. Use a logged-in persistent profile only for personal research.

Usage:
  pip install -r requirements.txt
  playwright install chromium

  # Headed browser (recommended first run so you can sign in):
  python scrape_linkedin_jobs.py --headed --user-data-dir ./.pw-linkedin-profile

  # Defaults: Data Engineer, Edison NJ, 50 mi, past week, 20 jobs
  python scrape_linkedin_jobs.py --out ../data/linkedin_jobs.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR.parent / "data" / "linkedin_jobs.json"

F_TPR = {"day": "r86400", "week": "r604800", "month": "r2592000"}

DISTANCE_RANGE_VALUE = {0: 0, 5: 1, 10: 2, 25: 3, 50: 4, 100: 5}


def clean_title(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    half = len(s) // 2
    if half > 0 and s[:half] == s[half:].strip():
        return s[:half].strip()
    return s


def search_url(keywords: str, location: str, distance_mi: int, when: str) -> str:
    f_tpr = F_TPR[when]
    return (
        "https://www.linkedin.com/jobs/search/?"
        f"keywords={quote_plus(keywords)}"
        f"&location={quote_plus(location)}"
        f"&distance={distance_mi}"
        f"&f_TPR={f_tpr}"
    )


def apply_distance_filter_if_needed(page, want_mi: int) -> None:
    """Clicks Distance filter, sets range step for 50 mi etc., applies results."""
    want = DISTANCE_RANGE_VALUE.get(want_mi)
    if want is None:
        raise ValueError(f"distance_mi must be one of {list(DISTANCE_RANGE_VALUE.keys())}")

    dist_btn = page.get_by_role("button", name=re.compile(r"Distance filter", re.I))
    if dist_btn.count() == 0:
        return
    try:
        dist_btn.first.click(timeout=5000)
    except PlaywrightTimeout:
        return

    slider = page.locator('input[type="range"]')
    if slider.count() == 0:
        return
    try:
        slider.first.evaluate(
            """(el, val) => {
                el.value = String(val);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }""",
            want,
        )
    except Exception:
        pass

    for name_pat in (
        re.compile(r"Show \d+\+ results", re.I),
        re.compile(r"Apply current filter", re.I),
    ):
        b = page.get_by_role("button", name=name_pat)
        if b.count():
            try:
                b.first.click(timeout=3000)
                page.wait_for_timeout(1500)
                return
            except PlaywrightTimeout:
                continue


def collect_jobs_from_list(page, max_ids: int, scroll_rounds: int = 120) -> list[dict]:
    """Scroll the virtualized list and gather unique job postings from list cards."""

    seen: dict[str, dict] = {}
    stable = 0
    prev = -1

    for _ in range(scroll_rounds):
        batch = page.evaluate(
            """() => {
                function cleanTitle(t) {
                  if (!t) return '';
                  const s = t.replace(/\\n/g, ' ').replace(/\\s+/g, ' ').trim();
                  const half = Math.floor(s.length / 2);
                  if (half > 0 && s.slice(0, half) === s.slice(half).trim()) return s.slice(0, half).trim();
                  return s;
                }
                const jobs = [];
                document.querySelectorAll('li.scaffold-layout__list-item').forEach((li) => {
                  const link = li.querySelector('a[href*="/jobs/view/"]');
                  if (!link) return;
                  const m = link.href.match(/jobs\\/view\\/(\\d+)/);
                  if (!m) return;
                  const id = m[1];
                  const titleEl = li.querySelector('.artdeco-entity-lockup__title');
                  const subEl = li.querySelector('.artdeco-entity-lockup__subtitle');
                  const capEl = li.querySelector('.artdeco-entity-lockup__caption');
                  const listTime = li.querySelector('time');
                  jobs.push({
                    job_id: id,
                    title: cleanTitle(titleEl && titleEl.innerText),
                    company: cleanTitle(subEl && subEl.innerText),
                    location: cleanTitle(capEl && capEl.innerText),
                    listed_time: (listTime && (listTime.innerText || listTime.getAttribute('datetime') || '')).trim(),
                    url: 'https://www.linkedin.com/jobs/view/' + id + '/',
                  });
                });
                return jobs;
            }"""
        )

        for row in batch:
            jid = row["job_id"]
            if jid not in seen:
                seen[jid] = row

        if len(seen) == prev:
            stable += 1
        else:
            stable = 0
        prev = len(seen)

        if len(seen) >= max_ids + 5 or stable >= 12:
            break

        page.evaluate(
            """() => {
                const sc = [...document.querySelectorAll('div')].find((d) => {
                  const st = getComputedStyle(d);
                  return (st.overflowY === 'auto' || st.overflowY === 'scroll')
                    && d.scrollHeight > d.clientHeight + 100;
                });
                if (sc) sc.scrollTop += 450;
            }"""
        )
        page.wait_for_timeout(350)

    return list(seen.values())[:max_ids]


def extract_about_the_job(page) -> str:
    return page.evaluate(
        """() => {
            const t = document.body.innerText;
            const m = t.match(/About the job\\s*\\n+([\\s\\S]*?)(?:\\n+Set alert for similar jobs|\\n+About the company|Trending employee)/i);
            let d = m ? m[1].trim() : '';
            const cut = d.split(/\\n\\nAbout Us\\n\\n/);
            return cut[0].trim();
        }"""
    )


def goto_next_results_page(page) -> bool:
    nxt = page.get_by_role("button", name=re.compile(r"View next page|Next page", re.I))
    if nxt.count() == 0:
        return False
    try:
        nxt.first.click(timeout=5000)
        page.wait_for_timeout(2000)
        return True
    except PlaywrightTimeout:
        return False


def scrape(
    keywords: str,
    location: str,
    distance_mi: int,
    when: str,
    max_jobs: int,
    headless: bool,
    user_data_dir: Path | None,
    slow_mo_ms: int,
    out_path: Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": "linkedin_jobs_search",
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "search": {
            "keywords": keywords,
            "location": location,
            "distance_mi": distance_mi,
            "date_posted": when,
            "f_TPR": F_TPR[when],
        },
        "jobs": [],
    }

    url = search_url(keywords, location, distance_mi, when)

    with sync_playwright() as p:
        context = None
        browser = None
        if user_data_dir:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(Path(user_data_dir).resolve()),
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
                slow_mo=slow_mo_ms,
            )
            page = context.pages[0] if context.pages else context.new_page()
        else:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
                slow_mo=slow_mo_ms,
            )
            context = browser.new_context()
            page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(4000)
        apply_distance_filter_if_needed(page, distance_mi)

        collected: list[dict] = []
        pages_walked = 0
        while len(collected) < max_jobs and pages_walked < 8:
            chunk = collect_jobs_from_list(page, max_jobs=max_jobs - len(collected) + 10)
            for row in chunk:
                if row["job_id"] not in {x["job_id"] for x in collected}:
                    collected.append(row)
                if len(collected) >= max_jobs:
                    break
            if len(collected) >= max_jobs:
                break
            if not goto_next_results_page(page):
                break
            pages_walked += 1

        collected = collected[:max_jobs]

        for i, row in enumerate(collected):
            page.goto(row["url"], wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(2000 + slow_mo_ms)
            try:
                descr = extract_about_the_job(page)
            except Exception:
                descr = ""
            payload["jobs"].append(
                {
                    **row,
                    "title_normalized": clean_title(row.get("title", "")),
                    "description": descr,
                    "description_extracted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            time.sleep(0.3)

        context.close()
        if browser is not None:
            browser.close()

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(payload['jobs'])} jobs to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keywords", default="Data Engineer")
    ap.add_argument("--location", default="Edison, NJ")
    ap.add_argument("--distance-mi", type=int, default=50)
    ap.add_argument("--days", dest="when", choices=["day", "week", "month"], default="week")
    ap.add_argument("--max-jobs", type=int, default=20)
    ap.add_argument("--headless", action="store_true", help="Run headless (often blocked until cookies exist).")
    ap.add_argument("--headed", action="store_true", help="Show browser window (default if neither head flag).")
    ap.add_argument(
        "--user-data-dir",
        type=Path,
        default=None,
        help="Persistent Chromium profile dir (stay logged in to LinkedIn).",
    )
    ap.add_argument("--slow-mo", type=int, default=0, help="Slow down ops by N ms (debug).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    headless = args.headless and not args.headed
    if not args.headless and not args.headed:
        headless = False

    scrape(
        keywords=args.keywords,
        location=args.location,
        distance_mi=args.distance_mi,
        when=args.when,
        max_jobs=args.max_jobs,
        headless=headless,
        user_data_dir=args.user_data_dir,
        slow_mo_ms=args.slow_mo,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
