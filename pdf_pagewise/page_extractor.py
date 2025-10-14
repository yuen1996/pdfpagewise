import io
import os
import json
import time
import re
import subprocess
import tempfile
import pathlib
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import ImageOps, ImageFilter
import pytesseract
from pytesseract import Output
from openai import OpenAI

from schema import (
    CanonicalDoc, DocSource, PageModel, TableModel, TableColumn, FieldValue
)

from postprocess import (
    normalize_first_schedule_items,
    normalize_second_schedule_items,
    clean_description,  # clean table cells
)

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

def _pdftotext_page(file_bytes: bytes, page_no: int) -> str:
    """Poppler fallback: extract one page's text with better spacing."""
    with tempfile.TemporaryDirectory() as td:
        pdf = pathlib.Path(td) / "in.pdf"
        out = pathlib.Path(td) / "out.txt"
        pdf.write_bytes(file_bytes)
        subprocess.run(
            ["pdftotext", "-f", str(page_no), "-l", str(page_no), "-layout", str(pdf), str(out)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return out.read_text(encoding="utf-8", errors="ignore") if out.exists() else ""

# ------- OCR preprocessing + multi-pass -------
def _preprocess_pil(img):
    """Grayscale, autocontrast, light denoise, binarize."""
    g = img.convert("L")
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # simple threshold; works well for CamScanner
    return g.point(lambda x: 255 if x > 180 else 0, mode="1").convert("L")

def _deskew_via_osd(img, psm="0"):
    """Use Tesseract OSD to detect rotation and rotate back."""
    try:
        osd = pytesseract.image_to_osd(img, config=f"--psm {psm}")
        for line in osd.splitlines():
            if line.lower().startswith("rotate:"):
                angle = int(line.split(":")[1].strip())
                if angle:
                    return img.rotate(-angle, expand=True)
    except Exception:
        pass
    return img

def _ocr_pass(img, langs, psm, oem):
    cfg = f"--psm {psm} --oem {oem}"
    data = pytesseract.image_to_data(img, lang=langs, config=cfg, output_type=Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if c not in (-1, "-1")]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    text = pytesseract.image_to_string(img, lang=langs, config=cfg)
    return text, mean_conf

def _ocr_best_of(img):
    """Run several language mixes and return the text with highest confidence."""
    langs_env = os.getenv("OCR_LANGS", "chi_sim+chi_tra+eng+msa")
    candidates = [
        os.getenv("OCR_LANGS_CHI", "chi_sim+chi_tra"),
        os.getenv("OCR_LANGS_LATIN", "eng+msa"),
        langs_env,  # combined
    ]
    psm = os.getenv("OCR_PSM", "6")
    oem = os.getenv("OCR_OEM", "1")
    best = ("", -1.0)
    for l in candidates:
        t, c = _ocr_pass(img, l, psm, oem)
        if c > best[1]:
            best = (t, c)
    return best[0]

def _page_text_with_ocr(pdf: pdfplumber.PDF, file_bytes: bytes, idx1: int) -> Tuple[str, bool]:
    """Extract text for 1-based page index. Try pdfplumber -> pdftotext -> OCR (with preprocessing & OSD)."""
    page = pdf.pages[idx1 - 1]
    ocr_used = False

    # 1) pdfplumber layer
    try:
        txt = page.extract_text() or ""
    except Exception:
        txt = ""

    # 2) Poppler fallback (often yields cleaner spaces)
    if len(txt.strip()) < 20:
        txt = _pdftotext_page(file_bytes, idx1) or txt

    if len(txt.strip()) >= 20:
        return txt, ocr_used

    # 3) OCR path
    dpi = int(os.getenv("OCR_DPI", "400"))
    imgs = convert_from_bytes(file_bytes, dpi=dpi, first_page=idx1, last_page=idx1)
    if not imgs:
        return txt or "", ocr_used

    img = imgs[0]
    img = _preprocess_pil(img)
    img = _deskew_via_osd(img, psm="0")
    ocr_text = _ocr_best_of(img)
    if ocr_text and len(ocr_text.strip()) > 0:
        ocr_used = True
        return ocr_text, ocr_used

    return txt or "", ocr_used
# ----------------------------------------------

def _is_first_schedule_header(header: List[str]) -> bool:
    """Heuristic: true if header looks like the First Schedule table."""
    if not header or len(header) < 3:
        return False
    h = [str(x or "").lower() for x in header[:3]]
    # accept variants like "heading (1)", "subheading (2)", "description (3)"
    return ("heading" in h[0]) and ("subheading" in h[1]) and ("description" in h[2])

def _nonempty_ratio(rows: List[List[Any]]) -> float:
    total = sum(len(r) for r in rows) or 1
    nonempty = 0
    for r in rows:
        for c in r:
            if c is not None and str(c).strip():
                nonempty += 1
    return nonempty / total

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

        # Keep Schedule-style tables (First: 3 headers; Second: any header containing Rate)
        keep = _is_first_schedule_header(header) or any("rate" in str(h or "").lower() for h in header)
        if not keep:
            continue

        # Junk/near-empty table filter (e.g., small notices)
        if _nonempty_ratio(rows) < 0.2:
            continue

        # Fix duplicate/blank headers
        if len(set((h or "") for h in header)) != len(header):
            header = [f"col_{i+1}" for i in range(len(rows[0]))]

        cols = [TableColumn(name=str(h or f"col_{i+1}"), type="string") for i, h in enumerate(header)]

        # CLEAN description column(s) so “Bu tterm ilk” -> “Buttermilk”
        cleaned_rows = []
        for r in rows:
            rr = [str(c) if c is not None else None for c in r]
            if len(rr) >= 3 and rr[2]:
                rr[2] = clean_description(rr[2])
            cleaned_rows.append(rr)

        frames.append(
            TableModel(
                name=f"p{idx1:02d}_tbl{j:02d}",
                page=idx1,
                columns=cols,
                rows=cleaned_rows,
            )
        )
    return frames

def _guess_lang_for_page(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "jadual pertama" in t or "perintah" in t:
        return "ms"
    if "first schedule" in t or "order" in t:
        return "en"
    if any("\u4e00" <= ch <= "\u9fff" for ch in t):  # crude Chinese check
        return "zh"
    return None

def _extract_first_schedule_rate_from_text(text: str) -> Optional[str]:
    """Look for '5 per cent' / 'five per centum' / '5%'. If found, return '5%'."""
    t = (text or "").lower()
    for p in [r"\b5\s*%\b", r"\bfive\s+per\s+cent(?:um)?\b", r"\b5\s+per\s+cent(?:um)?\b"]:
        if re.search(p, t):
            return "5%"
    return None

def _llm_extract_kv(text: str, fields: List[str], api_key: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    clipped = text[:MAX_CHARS_PER_PAGE]
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an information extraction engine. Return ONLY valid JSON. If a field is missing, use null."},
            {"role": "user", "content": "Extract the following fields from this page's text.\n"
                                        f"Fields: {fields}.\n\n{clipped}"},
        ],
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

        # pass 1: page text/tables
        for idx1 in range(1, total_pages + 1):
            txt, ocr_used = _page_text_with_ocr(pdf, file_bytes, idx1)
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

            pages.append(PageModel(
                page_index=idx1,
                text=txt,
                tables=tables,
                kv_fields=kv_fields,
                lang=_guess_lang_for_page(txt),
                ocr=ocr_used
            ))

    # Build top-level
    source = DocSource(filename=filename, pages=len(pages), sha256=sha, lang=lang)
    doc = CanonicalDoc(doc_id=doc_id, source=source, pages=pages)

    # Meta summary
    doc.meta = {
        "cover_text": (pages[0].text[:600] if pages else None),
        "language_pages": [{"page_index": p.page_index, "lang": p.lang} for p in pages if p.lang],
    }

    # Detect doc type & rules
    full_text = "\n".join(p.text for p in pages if p.text)
    if "first schedule" in full_text.lower():
        doc.doc_type = "sales_tax_schedule"
    rate = _extract_first_schedule_rate_from_text(full_text)
    if rate:
        doc.rules = (doc.rules or {})
        doc.rules["first_schedule_rate"] = rate

    # ---------- Normalized outputs ----------
    first_items = normalize_first_schedule_items(pages, rate, doc_id)   # includes rate_percent, hs_*, bullets
    second_items = normalize_second_schedule_items(pages, doc_id)       # includes currency, specific_rate_*, bullets

    if first_items or second_items:
        doc.normalized = (doc.normalized or {})
        if first_items:
            doc.normalized["first_schedule_items"] = first_items
        if second_items:
            doc.normalized["second_schedule_items"] = second_items

    return doc

def build_excel_sheets(doc: CanonicalDoc) -> Dict[str, "pandas.DataFrame"]:
    import pandas as pd
    sheets: Dict[str, pd.DataFrame] = {}

    # Text sheets per page
    for p in doc.pages:
        sname = f"p{p.page_index:02d}_text"[:31]
        df = pd.DataFrame({"text": [p.text]})
        df.insert(0, "lang", p.lang)
        df.insert(0, "ocr", p.ocr)
        sheets[sname] = df

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

    # Normalized sheets
    if doc.normalized and "first_schedule_items" in doc.normalized:
        n1 = pd.DataFrame(doc.normalized["first_schedule_items"])
        if "bullets" in n1.columns:
            n1["bullets_joined"] = n1["bullets"].apply(lambda x: ("\n• " + "\n• ".join(x)) if isinstance(x, list) and x else "")
        sheets["first_schedule_items"] = n1

    if doc.normalized and "second_schedule_items" in doc.normalized:
        n2 = pd.DataFrame(doc.normalized["second_schedule_items"])
        if "bullets" in n2.columns:
            n2["bullets_joined"] = n2["bullets"].apply(lambda x: ("\n• " + "\n• ".join(x)) if isinstance(x, list) and x else "")
        sheets["second_schedule_items"] = n2

    # Rules sheet (e.g., 5%)
    if doc.rules:
        r = [{"key": k, "value": v} for k, v in doc.rules.items()]
        sheets["rules"] = pd.DataFrame(r)

    # Meta sheet
    if doc.meta:
        m = [{"key": k, "value": (v if not isinstance(v, list) else json.dumps(v, ensure_ascii=False))} for k, v in doc.meta.items()]
        sheets["meta"] = pd.DataFrame(m)

    return sheets
