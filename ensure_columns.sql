-- ensure_columns.sql — idempotent, NON-destructive.
-- Run ONCE in the Supabase SQL editor BEFORE the first local ingest run.
--
-- It ONLY adds columns the local script writes, IF they are missing, and adds the
-- unique keys the upserts rely on. It never drops anything and never touches your
-- existing read RPCs (questions_text_search / knowledge_chunks_search /
-- simhash_lookup / phash_lookup).

create extension if not exists vector;

-- ── public.questions ──────────────────────────────────────────────────────────
-- VERIFIED already present: question_embedding(vector 768), solution_md,
-- explanation_md, confidence_score, confidence_reason, simhash_binary, options,
-- answer_letter/answer_text, subject/chapter/topic, keywords, has_figure, page.
-- So nothing to add here; we only ensure the upsert conflict key exists.
do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'questions_doc_qno_unique') then
    alter table public.questions add constraint questions_doc_qno_unique unique (source_doc, q_no);
  end if;
end $$;

-- ── public.knowledge_chunks (read by knowledge_chunks_search) ─────────────────
alter table public.knowledge_chunks add column if not exists embedding vector(768);
do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'knowledge_chunks_unique') then
    alter table public.knowledge_chunks
      add constraint knowledge_chunks_unique unique (source_doc, page_number, chunk_index);
  end if;
end $$;

-- ── public.question_cache (read by simhash_lookup / phash_lookup) ─────────────
alter table public.question_cache add column if not exists simhash_binary   text;
alter table public.question_cache add column if not exists confidence_score int default 100;
alter table public.question_cache add column if not exists diagram_url      text;
