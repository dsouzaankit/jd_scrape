-- Build latest/categorized/curated JD tables from append-only raw embeddings.
-- Run this before running transformations.sql crosswalk query.

ATTACH 'data/jds_books.duckdb' AS jds_books;
USE jds_books;

-- Build latest non-obsolete JD snapshot from append-only raw table.
CREATE OR REPLACE VIEW staging_jd_latest AS
SELECT * EXCLUDE (rn)
FROM (
    SELECT j.*,
           ROW_NUMBER() OVER (
               PARTITION BY j.chunk_id
               ORDER BY coalesce(j.source_scraped_at_utc, j.ingested_at_utc) DESC,
                        j.ingested_at_utc DESC,
                        j.run_id DESC
           ) AS rn
    FROM staging_jd_raw j
    WHERE coalesce(j.is_obsolete, FALSE) = FALSE
) t
WHERE rn = 1;

-- Incremental reconciliation buckets (source window vs bounded target window).
-- Use this block for debugging and validating anti-join/full-outer logic.
-- Expected params (edit literals as needed):
--   source horizon: last 2 days
--   target horizon: last 30 days
WITH source_window AS (
    SELECT *
    FROM staging_jd_raw
    WHERE coalesce(source_scraped_at_utc, ingested_at_utc) >= now() - INTERVAL 2 DAY
),
target_window AS (
    SELECT *
    FROM staging_jd_latest
    WHERE coalesce(source_scraped_at_utc, ingested_at_utc) >= now() - INTERVAL 30 DAY
),
source_bounds AS (
    SELECT
        min(coalesce(source_scraped_at_utc, ingested_at_utc)) AS source_min_ts,
        max(coalesce(source_scraped_at_utc, ingested_at_utc)) AS source_max_ts
    FROM source_window
),
joined AS (
    SELECT
        s.chunk_id AS s_chunk_id,
        t.chunk_id AS t_chunk_id,
        s.content_hash AS s_content_hash,
        t.content_hash AS t_content_hash,
        s.job_id AS s_job_id,
        t.job_id AS t_job_id,
        s.document AS s_document,
        t.document AS t_document,
        s.embedding AS s_embedding,
        t.embedding AS t_embedding,
        s.run_id AS s_run_id,
        t.run_id AS t_run_id,
        s.ingested_at_utc AS s_ingested_at_utc,
        t.ingested_at_utc AS t_ingested_at_utc,
        s.last_seen_at_utc AS s_last_seen_at_utc,
        t.last_seen_at_utc AS t_last_seen_at_utc,
        s.source_scraped_at_utc AS s_source_scraped_at_utc,
        t.source_scraped_at_utc AS t_source_scraped_at_utc,
        s.scrape_batch_date AS s_scrape_batch_date,
        t.scrape_batch_date AS t_scrape_batch_date,
        s.scrape_attempt_epoch AS s_scrape_attempt_epoch,
        t.scrape_attempt_epoch AS t_scrape_attempt_epoch
    FROM source_window s
    -- remove full outer target side??
    FULL OUTER JOIN target_window t
      ON s.chunk_id = t.chunk_id
),
classified AS (
    SELECT
        j.*,
        b.source_min_ts,
        b.source_max_ts,
        CASE
            WHEN j.s_chunk_id IS NOT NULL AND j.t_chunk_id IS NOT NULL AND coalesce(j.s_content_hash, '') <> coalesce(j.t_content_hash, '')
                THEN 'matched_changed'
            WHEN j.s_chunk_id IS NOT NULL AND j.t_chunk_id IS NOT NULL
                THEN 'matched_same'
            WHEN j.s_chunk_id IS NOT NULL AND j.t_chunk_id IS NULL
                THEN 'source_only'
            WHEN j.s_chunk_id IS NULL AND j.t_chunk_id IS NOT NULL
                 AND coalesce(j.t_source_scraped_at_utc, j.t_ingested_at_utc) BETWEEN b.source_min_ts AND b.source_max_ts
                THEN 'target_only_in_window'
            WHEN j.s_chunk_id IS NULL AND j.t_chunk_id IS NOT NULL
                THEN 'target_only_out_window'
            ELSE 'unclassified'
        END AS reconciliation_bucket
    FROM joined j
    CROSS JOIN source_bounds b
)
SELECT reconciliation_bucket, count(*) AS row_count
FROM classified
GROUP BY 1
ORDER BY 1;

-- Materialized reconciliation rows (SCD-style output candidate).
-- This returns the unioned dataset that you can INSERT OVERWRITE into a curated table.
CREATE OR REPLACE TABLE staging_jd_curated AS
WITH source_window AS (
    SELECT *
    FROM staging_jd_raw
    WHERE coalesce(source_scraped_at_utc, ingested_at_utc) >= now() - INTERVAL 2 DAY
),
target_window AS (
    SELECT *
    FROM staging_jd_latest
    WHERE coalesce(source_scraped_at_utc, ingested_at_utc) >= now() - INTERVAL 30 DAY
),
source_bounds AS (
    SELECT
        min(coalesce(source_scraped_at_utc, ingested_at_utc)) AS source_min_ts,
        max(coalesce(source_scraped_at_utc, ingested_at_utc)) AS source_max_ts
    FROM source_window
),
joined AS (
    SELECT
        s.chunk_id AS s_chunk_id,
        t.chunk_id AS t_chunk_id,
        s.content_hash AS s_content_hash,
        t.content_hash AS t_content_hash,
        s.job_id AS s_job_id,
        t.job_id AS t_job_id,
        s.document AS s_document,
        t.document AS t_document,
        s.embedding AS s_embedding,
        t.embedding AS t_embedding,
        s.chunk_index AS s_chunk_index,
        t.chunk_index AS t_chunk_index,
        s.num_chunks AS s_num_chunks,
        t.num_chunks AS t_num_chunks,
        s.title AS s_title,
        t.title AS t_title,
        s.company AS s_company,
        t.company AS t_company,
        s.location AS s_location,
        t.location AS t_location,
        s.url AS s_url,
        t.url AS t_url,
        s.requirements_headers AS s_requirements_headers,
        t.requirements_headers AS t_requirements_headers,
        s.similarity_to_requirements AS s_similarity_to_requirements,
        t.similarity_to_requirements AS t_similarity_to_requirements,
        s.has_requirements_sections AS s_has_requirements_sections,
        t.has_requirements_sections AS t_has_requirements_sections,
        s.embedding_model AS s_embedding_model,
        t.embedding_model AS t_embedding_model,
        s.run_id AS s_run_id,
        t.run_id AS t_run_id,
        s.source_json AS s_source_json,
        t.source_json AS t_source_json,
        s.ingested_at_utc AS s_ingested_at_utc,
        t.ingested_at_utc AS t_ingested_at_utc,
        s.last_seen_at_utc AS s_last_seen_at_utc,
        t.last_seen_at_utc AS t_last_seen_at_utc,
        s.source_scraped_at_utc AS s_source_scraped_at_utc,
        t.source_scraped_at_utc AS t_source_scraped_at_utc,
        s.scrape_batch_date AS s_scrape_batch_date,
        t.scrape_batch_date AS t_scrape_batch_date,
        s.scrape_attempt_epoch AS s_scrape_attempt_epoch,
        t.scrape_attempt_epoch AS t_scrape_attempt_epoch
    FROM source_window s
    FULL OUTER JOIN target_window t
      ON s.chunk_id = t.chunk_id
),
classified AS (
    SELECT
        j.*,
        b.source_min_ts,
        b.source_max_ts,
        CASE
            WHEN j.s_chunk_id IS NOT NULL AND j.t_chunk_id IS NOT NULL AND coalesce(j.s_content_hash, '') <> coalesce(j.t_content_hash, '')
                THEN 'matched_changed'
            WHEN j.s_chunk_id IS NOT NULL AND j.t_chunk_id IS NOT NULL
                THEN 'matched_same'
            WHEN j.s_chunk_id IS NOT NULL AND j.t_chunk_id IS NULL
                THEN 'source_only'
            WHEN j.s_chunk_id IS NULL AND j.t_chunk_id IS NOT NULL
                 AND coalesce(j.t_source_scraped_at_utc, j.t_ingested_at_utc) BETWEEN b.source_min_ts AND b.source_max_ts
                THEN 'target_only_in_window'
            WHEN j.s_chunk_id IS NULL AND j.t_chunk_id IS NOT NULL
                THEN 'target_only_out_window'
            ELSE 'unclassified'
        END AS reconciliation_bucket
    FROM joined j
    CROSS JOIN source_bounds b
),
unioned AS (
    -- Matched and changed: keep new source row active.
    SELECT
        c.s_chunk_id AS chunk_id,
        c.s_document AS document,
        c.s_embedding AS embedding,
        c.s_job_id AS job_id,
        c.s_chunk_index AS chunk_index,
        c.s_num_chunks AS num_chunks,
        c.s_title AS title,
        c.s_company AS company,
        c.s_location AS location,
        c.s_url AS url,
        c.s_requirements_headers AS requirements_headers,
        c.s_similarity_to_requirements AS similarity_to_requirements,
        c.s_has_requirements_sections AS has_requirements_sections,
        c.s_embedding_model AS embedding_model,
        c.s_content_hash AS content_hash,
        c.s_run_id AS run_id,
        c.s_source_json AS source_json,
        c.s_ingested_at_utc AS ingested_at_utc,
        coalesce(c.s_source_scraped_at_utc, c.s_ingested_at_utc) AS last_seen_at_utc,
        FALSE AS is_obsolete,
        NULL::TIMESTAMPTZ AS obsoleted_at_utc,
        c.s_scrape_batch_date AS scrape_batch_date,
        c.s_scrape_attempt_epoch AS scrape_attempt_epoch,
        c.s_source_scraped_at_utc AS source_scraped_at_utc,
        c.reconciliation_bucket
    FROM classified c
    WHERE c.reconciliation_bucket = 'matched_changed'

    UNION ALL

    -- Matched and changed: retain previous target row as obsolete.
    SELECT
        c.t_chunk_id AS chunk_id,
        c.t_document AS document,
        c.t_embedding AS embedding,
        c.t_job_id AS job_id,
        c.t_chunk_index AS chunk_index,
        c.t_num_chunks AS num_chunks,
        c.t_title AS title,
        c.t_company AS company,
        c.t_location AS location,
        c.t_url AS url,
        c.t_requirements_headers AS requirements_headers,
        c.t_similarity_to_requirements AS similarity_to_requirements,
        c.t_has_requirements_sections AS has_requirements_sections,
        c.t_embedding_model AS embedding_model,
        c.t_content_hash AS content_hash,
        c.t_run_id AS run_id,
        c.t_source_json AS source_json,
        c.t_ingested_at_utc AS ingested_at_utc,
        c.t_last_seen_at_utc AS last_seen_at_utc,
        TRUE AS is_obsolete,
        now()::TIMESTAMPTZ AS obsoleted_at_utc,
        c.t_scrape_batch_date AS scrape_batch_date,
        c.t_scrape_attempt_epoch AS scrape_attempt_epoch,
        c.t_source_scraped_at_utc AS source_scraped_at_utc,
        c.reconciliation_bucket
    FROM classified c
    WHERE c.reconciliation_bucket = 'matched_changed'

    UNION ALL

    -- Matched and same hash: keep target row, refresh last_seen_at from source time.
    SELECT
        c.t_chunk_id AS chunk_id,
        c.t_document AS document,
        c.t_embedding AS embedding,
        c.t_job_id AS job_id,
        c.t_chunk_index AS chunk_index,
        c.t_num_chunks AS num_chunks,
        c.t_title AS title,
        c.t_company AS company,
        c.t_location AS location,
        c.t_url AS url,
        c.t_requirements_headers AS requirements_headers,
        c.t_similarity_to_requirements AS similarity_to_requirements,
        c.t_has_requirements_sections AS has_requirements_sections,
        c.t_embedding_model AS embedding_model,
        c.t_content_hash AS content_hash,
        c.t_run_id AS run_id,
        c.t_source_json AS source_json,
        c.t_ingested_at_utc AS ingested_at_utc,
        coalesce(c.s_source_scraped_at_utc, c.s_ingested_at_utc, c.t_last_seen_at_utc) AS last_seen_at_utc,
        FALSE AS is_obsolete,
        NULL::TIMESTAMPTZ AS obsoleted_at_utc,
        c.t_scrape_batch_date AS scrape_batch_date,
        c.t_scrape_attempt_epoch AS scrape_attempt_epoch,
        c.t_source_scraped_at_utc AS source_scraped_at_utc,
        c.reconciliation_bucket
    FROM classified c
    WHERE c.reconciliation_bucket = 'matched_same'

    UNION ALL

    -- New source rows.
    SELECT
        c.s_chunk_id AS chunk_id,
        c.s_document AS document,
        c.s_embedding AS embedding,
        c.s_job_id AS job_id,
        c.s_chunk_index AS chunk_index,
        c.s_num_chunks AS num_chunks,
        c.s_title AS title,
        c.s_company AS company,
        c.s_location AS location,
        c.s_url AS url,
        c.s_requirements_headers AS requirements_headers,
        c.s_similarity_to_requirements AS similarity_to_requirements,
        c.s_has_requirements_sections AS has_requirements_sections,
        c.s_embedding_model AS embedding_model,
        c.s_content_hash AS content_hash,
        c.s_run_id AS run_id,
        c.s_source_json AS source_json,
        c.s_ingested_at_utc AS ingested_at_utc,
        coalesce(c.s_source_scraped_at_utc, c.s_ingested_at_utc) AS last_seen_at_utc,
        FALSE AS is_obsolete,
        NULL::TIMESTAMPTZ AS obsoleted_at_utc,
        c.s_scrape_batch_date AS scrape_batch_date,
        c.s_scrape_attempt_epoch AS scrape_attempt_epoch,
        c.s_source_scraped_at_utc AS source_scraped_at_utc,
        c.reconciliation_bucket
    FROM classified c
    WHERE c.reconciliation_bucket = 'source_only'

    UNION ALL

    -- Target-only rows inside source-covered window: mark obsolete.
    SELECT
        c.t_chunk_id AS chunk_id,
        c.t_document AS document,
        c.t_embedding AS embedding,
        c.t_job_id AS job_id,
        c.t_chunk_index AS chunk_index,
        c.t_num_chunks AS num_chunks,
        c.t_title AS title,
        c.t_company AS company,
        c.t_location AS location,
        c.t_url AS url,
        c.t_requirements_headers AS requirements_headers,
        c.t_similarity_to_requirements AS similarity_to_requirements,
        c.t_has_requirements_sections AS has_requirements_sections,
        c.t_embedding_model AS embedding_model,
        c.t_content_hash AS content_hash,
        c.t_run_id AS run_id,
        c.t_source_json AS source_json,
        c.t_ingested_at_utc AS ingested_at_utc,
        c.t_last_seen_at_utc AS last_seen_at_utc,
        TRUE AS is_obsolete,
        now()::TIMESTAMPTZ AS obsoleted_at_utc,
        c.t_scrape_batch_date AS scrape_batch_date,
        c.t_scrape_attempt_epoch AS scrape_attempt_epoch,
        c.t_source_scraped_at_utc AS source_scraped_at_utc,
        c.reconciliation_bucket
    FROM classified c
    WHERE c.reconciliation_bucket = 'target_only_in_window'

    UNION ALL

    -- Target-only rows outside source window: carry forward unchanged.
    SELECT
        c.t_chunk_id AS chunk_id,
        c.t_document AS document,
        c.t_embedding AS embedding,
        c.t_job_id AS job_id,
        c.t_chunk_index AS chunk_index,
        c.t_num_chunks AS num_chunks,
        c.t_title AS title,
        c.t_company AS company,
        c.t_location AS location,
        c.t_url AS url,
        c.t_requirements_headers AS requirements_headers,
        c.t_similarity_to_requirements AS similarity_to_requirements,
        c.t_has_requirements_sections AS has_requirements_sections,
        c.t_embedding_model AS embedding_model,
        c.t_content_hash AS content_hash,
        c.t_run_id AS run_id,
        c.t_source_json AS source_json,
        c.t_ingested_at_utc AS ingested_at_utc,
        c.t_last_seen_at_utc AS last_seen_at_utc,
        FALSE AS is_obsolete,
        NULL::TIMESTAMPTZ AS obsoleted_at_utc,
        c.t_scrape_batch_date AS scrape_batch_date,
        c.t_scrape_attempt_epoch AS scrape_attempt_epoch,
        c.t_source_scraped_at_utc AS source_scraped_at_utc,
        c.reconciliation_bucket
    FROM classified c
    WHERE c.reconciliation_bucket = 'target_only_out_window'
)
SELECT *
FROM unioned;

SELECT reconciliation_bucket, count(*) AS row_count
FROM staging_jd_curated
GROUP BY 1
ORDER BY 1;
