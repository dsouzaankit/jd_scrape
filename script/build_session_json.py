#!/usr/bin/env python3
"""Merge `data/descriptions_raw/<job_id>.txt` + metadata into one JSON export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "descriptions_raw"
OUT = ROOT / "data" / "linkedin_data_engineer_edison_50mi.json"

JOBS_META = [
    {
        "job_id": "4389202311",
        "title": "Data Engineer",
        "company": "Venteon",
        "location": "Brooklyn, NY (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4389202311/",
    },
    {
        "job_id": "4388992002",
        "title": "Senior Staff Data Engineer",
        "company": "Chobani",
        "location": "New York City Metropolitan Area (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4388992002/",
    },
    {
        "job_id": "4363131526",
        "title": "Senior Analytics Engineer",
        "company": "Fora Travel",
        "location": "New York, NY (On-site)",
        "url": "https://www.linkedin.com/jobs/view/4363131526/",
    },
    {
        "job_id": "4390386755",
        "title": "Data Engineer",
        "company": "Mindlance",
        "location": "Iselin, NJ (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4390386755/",
    },
    {
        "job_id": "4387978601",
        "title": "Data Engineer (Ad-Tech)",
        "company": "Genius Sports",
        "location": "New York, NY",
        "url": "https://www.linkedin.com/jobs/view/4387978601/",
    },
    {
        "job_id": "4335399466",
        "title": "Senior Data Engineer",
        "company": "Genentech",
        "location": "New York, NY",
        "url": "https://www.linkedin.com/jobs/view/4335399466/",
    },
    {
        "job_id": "4391777720",
        "title": "Sr./Lead AWS Data Engineer",
        "company": "FUSTIS LLC",
        "location": "Jersey City, NJ (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4391777720/",
    },
    {
        "job_id": "4387419689",
        "title": "Lead Data Engineer",
        "company": "Inizio Partners",
        "location": "New York City Metropolitan Area (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4387419689/",
    },
    {
        "job_id": "4392680263",
        "title": "Tech Lead (Python + Databricks)",
        "company": "Rivago Infotech Inc",
        "location": "Iselin, NJ (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4392680263/",
    },
    {
        "job_id": "4390361012",
        "title": "Senior Data Engineer",
        "company": "Svitla Systems, Inc.",
        "location": "NY, NJ or CT (on-site NY client)",
        "url": "https://www.linkedin.com/jobs/view/4390361012/",
    },
    {
        "job_id": "4391492440",
        "title": "Senior BI Data Engineer",
        "company": "Summit Financial",
        "location": "Parsippany, NJ (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4391492440/",
    },
    {
        "job_id": "4389326201",
        "title": "Data Engineer",
        "company": "Backstage",
        "location": "New York, NY (Remote)",
        "url": "https://www.linkedin.com/jobs/view/4389326201/",
    },
    {
        "job_id": "4390140692",
        "title": "Senior Data Engineer",
        "company": "LHH",
        "location": "United States (Remote)",
        "url": "https://www.linkedin.com/jobs/view/4390140692/",
    },
    {
        "job_id": "4391224312",
        "title": "Lead Data Engineer",
        "company": "Piper Companies",
        "location": "Fort Washington, PA (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4391224312/",
    },
    {
        "job_id": "4390111596",
        "title": "Data Engineer",
        "company": "ExpertsHub.ai",
        "location": "New Jersey, United States (Remote)",
        "url": "https://www.linkedin.com/jobs/view/4390111596/",
    },
    {
        "job_id": "4389971092",
        "title": "Senior Data Engineer",
        "company": "Burlington Stores, Inc.",
        "location": "Edgewater Park, NJ (Hybrid)",
        "url": "https://www.linkedin.com/jobs/view/4389971092/",
    },
    {
        "job_id": "4390657827",
        "title": "Senior Data Engineer",
        "company": "SMB Recruiting (client)",
        "location": "United States (Remote)",
        "url": "https://www.linkedin.com/jobs/view/4390657827/",
    },
    {
        "job_id": "4388967519",
        "title": "Data Engineer (Database & Backend)",
        "company": "HYGO",
        "location": "United States (Remote)",
        "url": "https://www.linkedin.com/jobs/view/4388967519/",
    },
    {
        "job_id": "4390356742",
        "title": "Senior Data Analytics Engineer",
        "company": "Optro",
        "location": "United States (Remote)",
        "url": "https://www.linkedin.com/jobs/view/4390356742/",
    },
    {
        "job_id": "4387975674",
        "title": "Senior Software Engineer (Data / ETL)",
        "company": "Genius Sports",
        "location": "New York, NY",
        "url": "https://www.linkedin.com/jobs/view/4387975674/",
    },
]


def main() -> None:
    jobs = []
    missing = []
    for meta in JOBS_META:
        jid = meta["job_id"]
        path = RAW_DIR / f"{jid}.txt"
        if not path.is_file():
            missing.append(jid)
            desc = ""
        else:
            desc = path.read_text(encoding="utf-8").strip()
        jobs.append({**meta, "description": desc})

    payload = {
        "source": "linkedin_jobs_search_mcp_session",
        "notes": "Captured 2026-03-28 via Chrome DevTools MCP; search: Data Engineer, Edison NJ, 50mi, past week.",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "search": {
            "keywords": "Data Engineer",
            "location": "Edison, NJ",
            "distance_mi": 50,
            "date_posted": "week",
            "f_TPR": "r604800",
        },
        "jobs": jobs,
        "missing_description_files": missing,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({len(jobs)} jobs, missing text for {len(missing)} ids)")


if __name__ == "__main__":
    main()
