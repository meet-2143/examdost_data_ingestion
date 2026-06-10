"""
sink.py — Local replacement for the n8n ingestion webhook.

The current pipeline does:   ingest_pdf.py  ──POST──▶  n8n webhook  ──▶  Supabase
n8n was the only piece doing embeddings + pHash + simHash + the Supabase writes.
This module re-implements *exactly that piece* locally, so the same parser output
lands in the same tables the live workflows (phase_1 / phase_2 / phase_3) read from.

Read contract (verified against the live `questions` table + the 3 workflow JSONs):
  questions_text_search   reads public.questions      (question_embedding vector(768),
                          solution_md, explanation_md, confidence_score, subject,
                          chapter, topic, answer_text, question_text, simhash_binary)
  knowledge_chunks_search reads public.knowledge_chunks (embedding vector(768),
                          chunk_text, subject, chapter, topic, page_number, q_no)
  simhash_lookup /        read  public.question_cache  (phash_col bytea,
  phash_lookup            simhash_binary text(64), solution_md, diagram_url,
                          confidence_score, subject, chapter, topic)

Nothing here talks to n8n.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from io import BytesIO
from typing import Optional

# ── Config (env-driven, same secrets that live in the workflow JSONs) ──────────
SUPABASE_URL   = os.environ.get("SUPABASE_URL",  "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_APIKEY", ())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

EMBED_MODEL    = "models/gemini-embedding-001"   # identical to phase_1
EMBED_DIM      = 768                              # identical to phase_1
EMBED_URL      = ("https://generativelanguage.googleapis.com/v1beta/"
                  f"{EMBED_MODEL}:embedContent?key={{key}}")

# Stored docs use RETRIEVAL_DOCUMENT; phase_1 uses RETRIEVAL_QUERY on the student's
# text. Same model + same dimensionality ⇒ comparable cosine distances.
EMBED_TASK_TYPE = "RETRIEVAL_DOCUMENT"

# Name of the vector column on public.questions (verified live).
QUESTIONS_VECTOR_COL = "question_embedding"

DEFAULT_CONFIDENCE = 100   # PDF answer-key questions are ground truth


# ===========================================================================
# Gemini embedding (768-d) — mirrors phase_1's "HTTP: RAG Embed Query" node
# ===========================================================================

def embed_text(text: str, *, retries: int = 4) -> Optional[list[float]]:
    """Return a 768-d embedding for `text`, or None on persistent failure."""
    text = (text or "").strip()
    if not text:
        return None
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    body = {
        "model": EMBED_MODEL,
        "content": {"parts": [{"text": text}]},
        "taskType": EMBED_TASK_TYPE,
        "outputDimensionality": EMBED_DIM,
    }
    url = EMBED_URL.format(key=GEMINI_API_KEY)
    data = json.dumps(body).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                out = json.loads(resp.read())
            values = (out.get("embedding") or {}).get("values")
            if values and len(values) == EMBED_DIM:
                return values
            print(f"[embed] unexpected response shape: {str(out)[:200]}", file=sys.stderr)
            return None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"[embed] {e.code}, retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"[embed] HTTP {e.code}: {e.read()[:200]}", file=sys.stderr)
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[embed] FAILED: {e}", file=sys.stderr)
            return None
    return None


def vector_literal(values: list[float]) -> str:
    """pgvector accepts a bracketed string literal via PostgREST."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


# ===========================================================================
# SimHash — byte-for-byte identical to phase_1's "Code: Normalize + SimHash"
# so a student's photo/text produces the SAME 64-bit string we store here.
# ===========================================================================

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower())).strip()


def simhash_binary(text: str, bits: int = 64) -> str:
    tokens = [t for t in _normalize(text).split(" ") if t]
    if not tokens:
        return "0" * bits
    vector = [0] * bits
    for token in tokens:
        h = 5381
        for ch in token:
            h = (((h << 5) + h) ^ ord(ch)) & 0xFFFFFFFF   # djb2, h>>>0
        for b in range(bits):
            vector[b] += 1 if (h >> (b % 32)) & 1 else -1
    return "".join("1" if v >= 0 else "0" for v in vector)


# ===========================================================================
# Perceptual hash (pHash) of a figure/page image → 8 bytes (bytea hex).
# Standard 32×32 → DCT → 8×8 low-freq → median threshold.
# ===========================================================================

def phash_hex(image_bytes: bytes) -> Optional[str]:
    try:
        from PIL import Image
        import numpy as np
        from scipy.fftpack import dct
    except Exception as e:
        print(f"[phash] deps missing ({e}); skipping pHash", file=sys.stderr)
        return None
    try:
        img = Image.open(BytesIO(image_bytes)).convert("L").resize((32, 32), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float64)
        d = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
        low = d[:8, :8]
        med = np.median(low[1:].flatten())   # exclude DC term
        bits = (low > med).flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return "\\x" + val.to_bytes(8, "big").hex()
    except Exception as e:
        print(f"[phash] failed: {e}", file=sys.stderr)
        return None


# ===========================================================================
# Supabase REST writers (service_role key bypasses RLS)
# ===========================================================================

def _headers() -> dict:
    if not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_KEY is not set")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _post(path: str, body, extra_headers: Optional[dict] = None) -> tuple[int, str]:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = _headers()
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def upsert_question(row: dict) -> tuple[int, str]:
    """Upsert one canonical question into public.questions (on source_doc,q_no)."""
    return _post(
        "questions?on_conflict=source_doc,q_no",
        [row],
        {"Prefer": "resolution=merge-duplicates,return=minimal"},
    )


def upsert_chunk(row: dict) -> tuple[int, str]:
    """Upsert one knowledge chunk (on source_doc,page_number,chunk_index)."""
    return _post(
        "knowledge_chunks?on_conflict=source_doc,page_number,chunk_index",
        [row],
        {"Prefer": "resolution=merge-duplicates,return=minimal"},
    )


def upsert_cache_figure(row: dict) -> tuple[int, str]:
    """Upsert one figure into public.question_cache (on phash_col)."""
    return _post(
        "question_cache?on_conflict=phash_col",
        [row],
        {"Prefer": "resolution=merge-duplicates,return=minimal"},
    )
