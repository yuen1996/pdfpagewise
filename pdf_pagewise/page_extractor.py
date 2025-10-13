import io
import os
import json
import time
from typing import List, Dict, Any, Optional

import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from openai import OpenAI

from schema import CanonicalDoc, DocSource, PageModel, TableModel, TableColumn, FieldValue

MAX_CHARS_PER_PAGE = 5000  # safety when calling LLM
DEFAULT_MODEL = "gpt-4o-mini"

# Optional Windows path for Tesseract (ignored on Ubuntu unless set)
_tess_cmd = os.getenv("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# ----------------
# Helpers
# ----------------
def _sha256_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _page_text_with_ocr(pdf: pdfplumber.PDF, file_bytes: bytes, idx1: int) -> str:
    """Extract text for 1-based page index; OCR fallback if little/no text."""
    page = pdf.pages[idx1 - 1]
    try:
        txt = page.extract_text() or ""
    except Exception:
        txt = ""

    if len(txt.strip()) >= 20:
        return txt

    # OCR this page only
    try:
        imgs = convert_from_bytes(file_bytes, dpi=300, first_page=idx1, last_page=idx1)
        if imgs:
            return pytesseract.image_to_string(imgs[0])
    except Exception:
        pass
    return txt or ""

def _extract_tables_for_page(page: pdfplumber.page.Page, idx1: int) -> List[TableModel]:
    frames: List[TableModel] = []
    try:
        tables = page.extract_tables() or []
    except Exception:
        tables = []
    for j, t in enumerate(tables, start=1):
        if not t or len(t) < 2:
            continue
        header, *rows = t
        if not header:
            continue
        # Fix duplicate/blank headers
        if len(set(h or "" for h in header)) != len(header):
            header = [f"col_{i+1}" for i in range(len(rows[0]))]
        cols = [TableColumn(name=str(h or f"col_{i+1}"), type="string") for i, h in enumerate(header)]
        frames.append(
            TableModel(
                name=f"p{idx1:02d}_tbl{j:02d}",
                page=idx1,
                columns=cols,
                rows=[[str(c) if c is not None else None for c in r] for r in rows],
            )
        )
    return frames

def _llm_extract_kv(text: str, fields: List[str], api_key: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    clipped = text[:MAX_CHARS_PER_PAGE]

    sys_msg = {
        "role": "system",
        "content": (
            "You are an information extraction engine. "
            "Return ONLY valid JSON. If a field is missing, use null."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            "Extract the following fields from this page's text.\n"
            f"Fields: {fields}.\n\n" + clipped
        ),
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[sys_msg, user_msg],
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {f: None for f in fields}

# ----------------
# Public API
# ----------------
def extract_pagewise(
    file_bytes: bytes,
    filename: str,
    *,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    fields: Optional[List[str]] = None,
    lang: Optional[str] = None,
) -> CanonicalDoc:
    sha = _sha256_bytes(file_bytes)
    ts = time.strftime("%Y%m%d_%H%M%S")
    doc_id = f"{ts}_{sha[:8]}"

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total_pages = len(pdf.pages)
        pages: List[PageModel] = []

        for idx1 in range(1, total_pages + 1):
            txt = _page_text_with_ocr(pdf, file_bytes, idx1)
            tables = _extract_tables_for_page(pdf.pages[idx1 - 1], idx1)

            kv_fields: Dict[str, FieldValue] = {}
            if fields and api_key:
                kv_raw = _llm_extract_kv(txt, fields, api_key, model=model)
                kv_fields = {
                    k: FieldValue(
                        value=v,
                        type=("number" if isinstance(v, (int, float)) else "string"),
                        page=idx1,
                    )
                    for k, v in kv_raw.items()
                }

            pages.append(PageModel(page_index=idx1, text=txt, tables=tables, kv_fields=kv_fields))

    source = DocSource(filename=filename, pages=len(pages), sha256=sha, lang=lang)
    doc = CanonicalDoc(doc_id=doc_id, source=source, pages=pages)
    return doc

def build_excel_sheets(doc: CanonicalDoc) -> Dict[str, "pandas.DataFrame"]:
    import pandas as pd
    sheets: Dict[str, pd.DataFrame] = {}

    # Text sheets
    for p in doc.pages:
        sname = f"p{p.page_index:02d}_text"[:31]
        sheets[sname] = pd.DataFrame({"text": [p.text]})

        # KV per page
        if p.kv_fields:
            rows = []
            for k, fv in p.kv_fields.items():
                rows.append({
                    "key": k,
                    "type": fv.type,
                    "unit": fv.unit,
                    "confidence": fv.confidence,
                    "page": fv.page,
                    "value": fv.value,
                })
            sheets[f"p{p.page_index:02d}_kv"[:31]] = pd.DataFrame(rows)

        # Tables per page
        for t in p.tables:
            cols = [c.name for c in t.columns]
            df = pd.DataFrame(t.rows, columns=cols)
            df.insert(0, "page", p.page_index)
            sheets[t.name[:31]] = df

    return sheets
