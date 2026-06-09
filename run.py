"""
run.py — Local end-to-end ingestion (no n8n).

Pipeline:
   PDF(s) ──▶ ingest_pdf.ingest()  ──▶  sink.py (embed + pHash + simHash + Supabase)
              (reused parser:                    (the job n8n used to do)
               text layer + Gemini OCR
               fallback for image PDFs)

The parser behaviour is IDENTICAL to the current workflow:
  * PDFs with a text layer  → regex parse (Question-N, options, Answer Key page).
  * Image / handwritten PDFs → pass --ocr-backend gemini --ocr-key <KEY>; pages
    with no text layer go through Gemini Vision, which also reads the visually
    marked correct answer. (Same code path the live ingest already uses.)

What lands in Supabase (consumable unchanged by phase_1/2/3):
  kind=question → public.questions          (question_embedding + solution_md + ...)
  kind=chunk    → public.knowledge_chunks    (embedding + chunk_text + hierarchy)
  kind=figure   → public.question_cache       (pHash + simHash + solution_md)   [--with-figures]

Run:
  # one text PDF
  python run.py --pdf "..\\03_Knowledge_Base\\...\\Lecture-05 Assignment Discussion.pdf" \
      --subject "Analog_Electronics" --chapter "Diode Circuits"

  # one image / handwritten PDF
  python run.py --pdf "...\\Lecture-17 Assignment Discussion_compressed.pdf" \
      --subject "Analog_Electronics" --chapter "Diode Circuits" --ocr-backend gemini

  # whole Knowledge Base tree (subject/chapter auto-derived from folder names)
  python run.py --root "..\\03_Knowledge_Base" --ocr-backend gemini --with-figures

  # preview only, no DB writes
  python run.py --root "..\\03_Knowledge_Base" --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reuse the existing, proven parser from <project root>/ingest
HERE = Path(__file__).resolve().parent          # ...\Rag_examdost\Doc Ingestion
ROOT = HERE.parent                               # ...\Rag_examdost
sys.path.insert(0, str(ROOT / "ingest"))
from ingest_pdf import ingest                      # noqa: E402
from ingest_multi import resolve_meta, load_folder_map, collect_pdfs  # noqa: E402

import sink                                         # noqa: E402


# ---------------------------------------------------------------------------
# Row builders — map parser payloads → exact columns the workflows read
# ---------------------------------------------------------------------------

def _question_row(p: dict):
    md = p.get("explanation_md") or ""
    emb = sink.embed_text(p.get("question_text") or "")
    row = {
        "source_doc":        p["source_doc"],
        "q_no":              p["q_no"],
        "page":              p.get("page"),
        "question_text":     p.get("question_text") or "",
        "options":           p.get("options") or {},
        "answer_letter":     p.get("answer_letter"),
        "answer_text":       p.get("answer_text"),
        # phase_1 reads solution_md first, explanation_md as fallback — set both.
        "solution_md":       md,
        "explanation_md":    md,
        "confidence_score":  sink.DEFAULT_CONFIDENCE,
        "confidence_reason": "pdf-ingest: answer-key verified",
        "subject":           p.get("subject"),
        "chapter":           p.get("chapter"),
        "topic":             p.get("topic"),
        "keywords":          p.get("keywords") or [],
        "has_figure":        bool(p.get("has_figure")),
        "simhash_binary":    sink.simhash_binary(p.get("question_text") or ""),
    }
    if emb is not None:
        row[sink.QUESTIONS_VECTOR_COL] = sink.vector_literal(emb)   # question_embedding
    return row, (emb is not None)


def _chunk_row(p: dict):
    emb = sink.embed_text(p.get("chunk_text") or "")
    row = {
        "source_doc":  p["source_doc"],
        "kind":        p.get("chunk_kind", "lecture"),
        "subject":     p.get("subject"),
        "chapter":     p.get("chapter"),
        "topic":       p.get("topic"),
        "q_no":        p.get("q_no"),
        "page_number": p.get("page_number"),
        "chunk_index": p.get("chunk_index", 0),
        "chunk_text":  p.get("chunk_text") or "",
    }
    if emb is not None:
        row["embedding"] = sink.vector_literal(emb)
    return row, (emb is not None)


def _figure_row(p: dict):
    import base64
    b64 = p.get("figure_image_b64")
    if not b64:
        return None
    ph = sink.phash_hex(base64.b64decode(b64))
    if not ph:
        return None
    return {
        "phash_col":        ph,
        "simhash_binary":   sink.simhash_binary(p.get("question_text") or ""),
        "solution_md":      p.get("fallback_solution_md") or "",
        "confidence_score": p.get("confidence", sink.DEFAULT_CONFIDENCE),
        "source_tag":       p.get("source_tag", "pdf-ingest"),
        "subject":          p.get("subject"),
        "chapter":          p.get("chapter"),
        "topic":            p.get("topic"),
    }


# ---------------------------------------------------------------------------
# Push one PDF's payloads
# ---------------------------------------------------------------------------

def push_payloads(payloads: list[dict], *, with_figures: bool, dry_run: bool, stats: dict) -> None:
    for p in payloads:
        kind = p.get("kind")

        if kind == "question":
            row, ok = _question_row(p)
            stats["question"] += 1
            if not ok:
                stats["embed_fail"] += 1
            if dry_run:
                print(f"  [dry] question q{p['q_no']} emb={'ok' if ok else 'MISS'} "
                      f"solution_md={len(row['solution_md'])}c", file=sys.stderr)
                continue
            code, msg = sink.upsert_question(row)
            _log("question", f"q{p['q_no']}", code, msg, stats)

        elif kind == "chunk":
            row, ok = _chunk_row(p)
            stats["chunk"] += 1
            if not ok:
                stats["embed_fail"] += 1
            if dry_run:
                continue
            code, msg = sink.upsert_chunk(row)
            _log("chunk", f"p{p.get('page_number')}#{p.get('chunk_index')}", code, msg, stats)

        elif kind == "figure":
            if not with_figures:
                continue
            row = _figure_row(p)
            stats["figure"] += 1
            if row is None:
                stats["phash_fail"] += 1
                continue
            if dry_run:
                continue
            code, msg = sink.upsert_cache_figure(row)
            _log("figure", f"q{p.get('q_no')}", code, msg, stats)


def _log(kind: str, ident: str, code: int, msg: str, stats: dict) -> None:
    ok = 200 <= code < 300
    stats["ok" if ok else "err"] += 1
    flag = "OK " if ok else f"ERR{code}"
    tail = "" if ok else f" :: {msg[:160]}"
    print(f"  [{flag}] {kind} {ident}{tail}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Local PDF → Supabase ingestion (no n8n).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",  help="Single PDF to ingest")
    src.add_argument("--root", help="Folder tree of PDFs (recursively)")

    ap.add_argument("--out-dir", default=str(HERE / "out"),
                    help="Where ingest_pdf writes manifest.json + extracted images")
    ap.add_argument("--subject", default=None, help="Override subject for all PDFs")
    ap.add_argument("--chapter", default=None, help="Override chapter for all PDFs")
    ap.add_argument("--folder-map", default=None, help="JSON map of folder→{subject,chapter}")
    ap.add_argument("--ocr-backend", choices=["none", "gemini"], default="none",
                    help="Use 'gemini' for image-only / handwritten PDFs")
    ap.add_argument("--ocr-key", default=None,
                    help="Gemini key for OCR (defaults to GEMINI_API_KEY env)")
    ap.add_argument("--with-figures", action="store_true",
                    help="Also upsert embedded figures into question_cache (pHash + simHash)")
    ap.add_argument("--dry-run", action="store_true", help="Parse + embed, but no DB writes")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N PDFs (0 = all)")
    args = ap.parse_args()

    ocr_key = args.ocr_key or os.environ.get("GEMINI_API_KEY") or sink.GEMINI_API_KEY
    if args.ocr_backend == "gemini" and not ocr_key:
        print("[warn] --ocr-backend gemini set but no key (pass --ocr-key or set GEMINI_API_KEY); "
              "image-only pages will be skipped.", file=sys.stderr)

    if not args.dry_run:
        if not sink.SUPABASE_KEY:
            print("[fatal] SUPABASE_SERVICE_KEY not set.", file=sys.stderr); return 2
        if not sink.GEMINI_API_KEY:
            print("[fatal] GEMINI_API_KEY not set (needed for embeddings).", file=sys.stderr); return 2

    out_dir = Path(args.out_dir)
    stats = {k: 0 for k in
             ("question", "chunk", "figure", "ok", "err", "embed_fail", "phash_fail")}

    # Build the (pdf, subject, chapter, out_dir) work list
    jobs: list[tuple[Path, str | None, str | None, Path]] = []
    if args.pdf:
        pdf = Path(args.pdf)
        if not pdf.exists():
            print(f"[fatal] PDF not found: {pdf}", file=sys.stderr); return 2
        jobs.append((pdf, args.subject, args.chapter, out_dir / pdf.stem))
    else:
        root = Path(args.root)
        if not root.is_dir():
            print(f"[fatal] --root is not a directory: {root}", file=sys.stderr); return 2
        fmap = load_folder_map(args.folder_map)
        for pdf in collect_pdfs(root):
            subj, chap = resolve_meta(pdf, root, fmap, args.subject, args.chapter)
            rel_parent = pdf.parent.relative_to(root)
            jobs.append((pdf, subj, chap, out_dir / rel_parent / pdf.stem))

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"[run] {len(jobs)} PDF(s) | dry_run={args.dry_run} | figures={args.with_figures}",
          file=sys.stderr)

    for i, (pdf, subj, chap, pdf_out) in enumerate(jobs, 1):
        print(f"\n[run] ({i}/{len(jobs)}) {pdf.name}  subject={subj!r} chapter={chap!r}",
              file=sys.stderr)
        try:
            payloads = ingest(pdf, pdf_out, subj, chap, args.ocr_backend, ocr_key)
        except Exception as e:
            print(f"[run] PARSE ERROR {pdf.name}: {e}", file=sys.stderr)
            continue
        push_payloads(payloads, with_figures=args.with_figures,
                      dry_run=args.dry_run, stats=stats)

    print("\n[run] ═══ Summary ═══════════════════════════════", file=sys.stderr)
    for k in ("question", "chunk", "figure"):
        print(f"[run]  {k:9}: {stats[k]}", file=sys.stderr)
    print(f"[run]  writes ok : {stats['ok']}   errors: {stats['err']}", file=sys.stderr)
    print(f"[run]  embed miss: {stats['embed_fail']}   phash miss: {stats['phash_fail']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
