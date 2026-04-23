-- Repo layout: run DuckDB with cwd = jd_scrape so data/ resolves, or use an absolute path.
ATTACH 'data/jds_books.duckdb' AS jds_books;
USE jds_books;

.mode line

-- Crosswalk JD vs book chunks. Embeddings are stored as LIST/unsized arrays; DuckDB's
-- array_cosine_similarity wants fixed ARRAY types. Use your model dim (384 for MiniLM-L6-v2).
-- If ingest used only float32, try FLOAT[384] instead of DOUBLE[384].
-- With normalize_embeddings=True, array_inner_product on unit vectors equals cosine similarity.
with book_info as (
    select book_title as book_title,
    section_index as book_section_index,
    first_value(section_title) over (
        partition by chapter_number, coalesce(isbn, book_title)
        order by section_index) as book_chapter_title,
    section_title as book_section_title_chunk_debug,
    chapter_number as book_chapter_number,
    -- left(document, 100) as book_chunk_1st100chars,
    chunk_id as book_chunk_id,
    embedding as book_embedding
    from staging_books
)
select
    -- jds.requirements_headers as jd_requirements_headers,
    jds.chunk_id as jd_chunk_id,
    left(jds.document, 100) as jd_chunk_text_1st100chars,
    right(jds.document, 100) as jd_chunk_text_last100chars,
    -- book_info.book_chunk_1st100chars as book_chunk_1st100chars,
    book_info.book_section_title_chunk_debug as book_section_title_chunk_debug,
    book_info.book_title as book_title,
    book_info.book_chapter_number as book_chapter_number,
    book_info.book_chapter_title as book_chapter_title,
    -- book_info.book_section_index as book_section_index,
    -- jds.requirements_headers as jd_requirements_headers,
    -- books.document as book_chunk_text,
    array_cosine_similarity(
        jds.embedding::DOUBLE[384],
        book_info.book_embedding::DOUBLE[384]
    ) as cosine_similarity
    , book_info.book_chunk_id as book_chunk_id
from staging_jd jds, book_info
where jds.requirements_headers != jds.document  -- ignore matches to requirement related section headers in jd
and jds.requirements_headers not like '%' || jds.document || '%'    -- ignore matches to requiremnent related section headers in jd
and jds.requirements_headers not like '%' || book_info.book_section_title_chunk_debug || '%'    -- ignore matches to requiremnent related section headers in jd
-- and lower(jds.document) not in ('requirements', 'work authorization', 'equal opportunity', 'eeo')
and not regexp_matches(jds.document, 'requirements|work authorization|authorized|equal opportunity|eeo|disability|401k|dental', 'i')
and not regexp_matches(jds.document, 'qualifications|values|benefits|compensation|salary|pay range|opportunity|position|role', 'i')
and array_cosine_similarity(
        jds.embedding::DOUBLE[384],
        book_info.book_embedding::DOUBLE[384]
    ) > 0.5
-- and jds.chunk_id like '4335399466%'
-- and jds.chunk_id like '4363131526%'
-- and jds.chunk_id like '4387419689%'
-- and jds.chunk_id like '4387975674%'
-- and jds.chunk_id like '4387978601%'
qualify rank() over (partition by jds.chunk_id order by cosine_similarity desc) <= 3
order by jds.chunk_id, cosine_similarity desc
limit 10
;



-- select chapter_number, section_index, section_title
-- from staging_books
-- where book_title like 'The Data Warehouse Toolkit%'
-- and section_index = 1
-- order by chapter_number, section_index
-- ;

-- delete from staging_books where book_title like 'The Data Warehouse Toolkit%';



/*
-- Example: attach another DB for one-off experiments (use paths that exist locally)
-- ATTACH 'other.duckdb' AS tmp_db;
-- USE tmp_db;
-- DETACH jds_books;
*/

