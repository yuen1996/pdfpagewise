import io
import os
import json
import time
import re
import subprocess
import tempfile
import pathlib
from typing import List, Dict, Any, Optional, Tuple, Callable

import pdfplumber
from pdf2image import convert_from_bytes

# OpenCV & Tesseract
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from PIL import Image
from openai import OpenAI

from schema import (
    CanonicalDoc, DocSource, PageModel, TableModel, TableColumn, FieldValue
)
from postprocess import (
    normalize_first_schedule_items,
    normalize_second_schedule_items,
    clean_description,
)

MAX_CHARS_PER_PAGE = 5000
DEFAULT_MODEL = "gpt-4o-mini"

_tess_cmd = os.getenv("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# ---------------- helpers ----------------
def _sha256_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(b)
    return h.hexdigest()

def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)

# ---------- OCRmyPDF (full-file, once) ----------
def _ocrmypdf_full(file_bytes: bytes, langs: str) -> Optional[bytes]:
    """OCR the entire PDF once; return OCR'd PDF bytes, or None if ocrmypdf missing/fails."""
    if not _which("ocrmypdf"):
        return None
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "src.pdf"
        dst = pathlib.Path(td) / "ocr.pdf"
        src.write_bytes(file_bytes)
        cmd = [
            "ocrmypdf", "--force-ocr",
            "--language", langs.replace("+", ","),
            "--optimize", "0",
            "--rotate-pages",
            str(src), str(dst)
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not dst.exists():
            return None
        return dst.read_bytes()

# ---------- poppler fallback per-page ----------
def _pdftotext_page(file_bytes: bytes, page_no: int) -> str:
    if not _which("pdftotext"):
        return ""
    with tempfile.TemporaryDirectory() as td:
        pdf = pathlib.Path(td) / "in.pdf"
        out = pathlib.Path(td) / "out.txt"
        pdf.write_bytes(file_bytes)
        subprocess.run(
            ["pdftotext", "-f", str(page_no), "-l", str(page_no), "-layout", str(pdf), str(out)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return out.read_text(encoding="utf-8", errors="ignore") if out.exists() else ""

# ---------------- OpenCV preprocess ----------------
def _opencv_preprocess(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    thr = _remove_outer_border(thr)
    thr = _deskew_by_hough(thr)
    return Image.fromarray(thr)

def _remove_outer_border(bin_img: np.ndarray) -> np.ndarray:
    h, w = bin_img.shape[:2]
    pad = max(5, int(0.01*min(h,w)))
    ff = bin_img.copy()
    mask = np.zeros((h+2,w+2), np.uint8)
    cv2.floodFill(ff, mask, (0,0), 255)
    merged = cv2.bitwise_or(bin_img, ff)
    coords = cv2.findNonZero(255-merged)
    if coords is not None:
        x,y,ww,hh = cv2.boundingRect(coords)
        x=max(0,x-pad); y=max(0,y-pad)
        ww=min(w-x, ww+2*pad); hh=min(h-y, hh+2*pad)
        merged = merged[y:y+hh, x:x+ww]
    return merged

def _deskew_by_hough(bin_img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angle = 0.0
    if lines is not None:
        angles = []
        for l in lines[:100]:
            _, theta = l[0]
            deg = theta*180/np.pi
            if 80 < deg < 100:
                continue
            if 0 <= deg < 45:
                angles.append(deg)
            elif 135 < deg <= 180:
                angles.append(deg-180)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) > 0.5:
        h, w = bin_img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        bin_img = cv2.warpAffine(bin_img, M, (w,h), flags=cv2.INTER_LINEAR, borderValue=255)
    return bin_img

# ---------------- OCR passes ----------------
def _tesseract_pass(pil_img: Image.Image, langs: str, psm: str, oem: str) -> Tuple[str, float]:
    cfg = f"--psm {psm} --oem {oem}"
    np_img = np.array(pil_img)
    data = pytesseract.image_to_data(np_img, lang=langs, config=cfg, output_type=Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if c not in (-1, "-1")]
    mean_conf = (sum(confs)/len(confs)) if confs else 0.0
    text = pytesseract.image_to_string(np_img, lang=langs, config=cfg)
    return text, float(mean_conf)

def _looks_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)

def _ocr_best_of(
    pil_img: Image.Image,
    *,
    speed_mode: bool,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    lang_hint_text: str = ""
) -> str:
    # Choose langs/psms aggressively but minimal in speed_mode
    is_cjk = _looks_cjk(lang_hint_text)
    langs_primary = "chi_sim+chi_tra" if is_cjk else "eng+msa"
    langs_combo   = os.getenv("OCR_LANGS", "chi_sim+chi_tra+eng+msa")

    psms = ["6","4"] if speed_mode else os.getenv("OCR_PSMS","6,4,11").split(",")
    oem = os.getenv("OCR_OEM","1")

    best = ("", -1.0)

    def try_one(L, P):
        nonlocal best
        if progress_cb: progress_cb({"stage":"ocr_try", "langs":L, "psm":P})
        t, c = _tesseract_pass(pil_img, L, P, oem)
        score = (c, len(t.strip()))
        if score > (best[1], len(best[0].strip())):
            best = (t, c)

    # First: primary langs only
    for psm in psms:
        try_one(langs_primary, psm)

    # Only if still poor, try combo (slower)
    if len(best[0].strip()) < 40 and not speed_mode:
        for psm in psms:
            try_one(langs_combo, psm)

    return best[0]

# ---------------- table extraction ----------------
PLUMBER_SETTINGS = [
    {"vertical_strategy": "lines", "horizontal_strategy": "lines",
     "intersection_x_tolerance": 5, "intersection_y_tolerance": 5},
    {"vertical_strategy": "text", "horizontal_strategy": "text",
     "snap_tolerance": 3, "join_tolerance": 3, "edge_min_length": 50},
    {"vertical_strategy": "lines", "horizontal_strategy": "text",
     "snap_tolerance": 3, "join_tolerance": 3},
]

def _is_first_schedule_header(header: List[str]) -> bool:
    if not header or len(header) < 3: return False
    h = [str(x or "").lower() for x in header[:3]]
    return ("heading" in h[0]) and ("subheading" in h[1]) and ("description" in h[2])

def _nonempty_ratio(rows: List[List[Any]]) -> float:
    total = sum(len(r) for r in rows) or 1
    nonempty = sum(1 for r in rows for c in r if c is not None and str(c).strip())
    return nonempty / total

def _extract_tables_for_page(page: pdfplumber.page.Page, idx1: int) -> List[TableModel]:
    frames: List[TableModel] = []
    tables = []
    seen = set()
    for sett in PLUMBER_SETTINGS:
        try:
            tbs = page.extract_tables(table_settings=sett) or []
        except Exception:
            tbs = []
        for t in tbs:
            if not t or len(t) < 2: continue
            header, *rows = t
            sig = tuple((str(x or "") for x in (header or [])))
            if sig in seen: continue
            seen.add(sig)
            tables.append((header, rows))

    for j, (header, rows) in enumerate(tables, start=1):
        if not header: continue
        keep = _is_first_schedule_header(header) or any("rate" in str(h or "").lower() for h in header)
        if not keep: continue
        if _nonempty_ratio(rows) < 0.2: continue

        if len(set((h or "") for h in header)) != len(header):
            header = [f"col_{i+1}" for i in range(len(rows[0]))]

        cols = [TableColumn(name=str(h or f"col_{i+1}"), type="string") for i, h in enumerate(header)]

        cleaned_rows = []
        for r in rows:
            rr = [str(c) if c is not None else None for c in r]
            if len(rr) >= 3 and rr[2]:
                rr[2] = clean_description(rr[2])
            cleaned_rows.append(rr)

        frames.append(TableModel(
            name=f"p{idx1:02d}_tbl{j:02d}",
            page=idx1,
            columns=cols,
            rows=cleaned_rows,
        ))
    return frames

def _guess_lang_for_page(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "jadual pertama" in t or "perintah" in t: return "ms"
    if "first schedule" in t or "order" in t: return "en"
    if any("\u4e00" <= ch <= "\u9fff" for ch in t): return "zh"
    return None

def _extract_first_schedule_rate_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    for p in [r"\b5\s*%\b", r"\bfive\s+per\s+cent(?:um)?\b", r"\b5\s+per\s+cent(?:um)?\b"]:
        if re.search(p, t): return "5%"
    return None

def _llm_extract_kv(text: str, fields: List[str], api_key: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    clipped = text[:MAX_CHARS_PER_PAGE]
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":"You are an information extraction engine. Return ONLY valid JSON. If a field is missing, use null."},
            {"role":"user","content": "Extract the following fields from this page's text.\n"
                                      f"Fields: {fields}.\n\n{clipped}"},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {f: None for f in fields}

# ---------------- main API ----------------
def extract_pagewise(
    file_bytes: bytes,
    filename: str,
    *,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    fields: Optional[List[str]] = None,
    lang: Optional[str] = None,
    aggressive_ocr: bool = True,
    try_ocrmypdf: bool = True,
    speed_mode: bool = True,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CanonicalDoc:

    sha = _sha256_bytes(file_bytes)
    ts = time.strftime("%Y%m%d_%H%M%S")
    doc_id = f"{ts}_{sha[:8]}"

    # (A) open original
    base_pdf = pdfplumber.open(io.BytesIO(file_bytes))

    # (B) run OCRmyPDF once (if needed) & reuse
    ocr_langs = os.getenv("OCR_LANGS", "chi_sim+chi_tra+eng+msa")
    ocr_pdf_obj = None
    if try_ocrmypdf:
        sample_len = 0
        for i in range(1, min(4, len(base_pdf.pages)) + 1):
            try:
                sample_len += len((base_pdf.pages[i-1].extract_text() or "").strip())
            except Exception:
                pass
        if sample_len < 80:
            if progress_cb: progress_cb({"stage":"ocrmypdf", "i":1, "n":len(base_pdf.pages)})
            ocr_bytes = _ocrmypdf_full(file_bytes, ocr_langs)
            if ocr_bytes:
                ocr_pdf_obj = pdfplumber.open(io.BytesIO(ocr_bytes))

    total_pages = len(base_pdf.pages)
    pages: List[PageModel] = []

    def _text_layer(pdf_obj: pdfplumber.PDF, i: int) -> str:
        try:
            return pdf_obj.pages[i-1].extract_text() or ""
        except Exception:
            return ""

    for idx1 in range(1, total_pages + 1):
        page_start = time.perf_counter()
        if progress_cb: progress_cb({"stage":"page_start", "i":idx1, "n":total_pages})

        # 1) text layer (prefer OCR'd pdf if present)
        txt = _text_layer(ocr_pdf_obj or base_pdf, idx1)
        # 2) pdftotext quick fallback
        if len(txt.strip()) < 20:
            if progress_cb: progress_cb({"stage":"pdftotext", "i":idx1, "n":total_pages})
            txt = _pdftotext_page(file_bytes, idx1) or txt

        ocr_used = False

        # 3) Tesseract OCR only if still weak
        if len(txt.strip()) < 60:
            dpi = int(os.getenv("OCR_DPI", "300" if speed_mode else "400"))
            imgs = convert_from_bytes(file_bytes, dpi=dpi, first_page=idx1, last_page=idx1)
            if imgs:
                if progress_cb: progress_cb({"stage":"preprocess", "i":idx1, "n":total_pages})
                pil = imgs[0]
                pil_bin = _opencv_preprocess(pil) if aggressive_ocr else pil
                if progress_cb: progress_cb({"stage":"ocr", "i":idx1, "n":total_pages})
                txt2 = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text=txt)
                if len(txt2.strip()) > len(txt.strip()):
                    txt = txt2
                    ocr_used = True

        # tables
        if progress_cb: progress_cb({"stage":"tables", "i":idx1, "n":total_pages})
        tables = _extract_tables_for_page((ocr_pdf_obj or base_pdf).pages[idx1 - 1], idx1)

        # AI fields (optional)
        kv_fields: Dict[str, FieldValue] = {}
        if fields and api_key:
            if progress_cb: progress_cb({"stage":"ai", "i":idx1, "n":total_pages})
            kv_raw = _llm_extract_kv(txt, fields, api_key, model=model)
            kv_fields = {
                k: FieldValue(value=v, type=("number" if isinstance(v, (int, float)) else "string"), page=idx1)
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

        if progress_cb:
            progress_cb({"stage":"page_done", "i":idx1, "n":total_pages, "sec": time.perf_counter()-page_start})

    # close PDFs
    try: base_pdf.close()
    except Exception: pass
    try:
        if ocr_pdf_obj: ocr_pdf_obj.close()
    except Exception: pass

    source = DocSource(filename=filename, pages=len(pages), sha256=sha, lang=lang)
    doc = CanonicalDoc(doc_id=doc_id, source=source, pages=pages)

    # doc-level summaries/normalization
    doc.meta = {
        "cover_text": (pages[0].text[:600] if pages else None),
        "language_pages": [{"page_index": p.page_index, "lang": p.lang} for p in pages if p.lang],
    }

    full_text = "\n".join(p.text for p in pages if p.text)
    if "first schedule" in full_text.lower():
        doc.doc_type = "sales_tax_schedule"
    rate = _extract_first_schedule_rate_from_text(full_text)
    if rate:
        doc.rules = (doc.rules or {}); doc.rules["first_schedule_rate"] = rate

    first_items = normalize_first_schedule_items(pages, rate, doc_id)
    second_items = normalize_second_schedule_items(pages, doc_id)
    if first_items or second_items:
        doc.normalized = (doc.normalized or {})
        if first_items:  doc.normalized["first_schedule_items"] = first_items
        if second_items: doc.normalized["second_schedule_items"] = second_items

    return doc

# ---------------- Excel sheet builder (needed by streamlit_app) ----------------
def build_excel_sheets(doc: CanonicalDoc) -> dict:
    """
    Build a dict of pandas DataFrames to write into an Excel file:
      - pXX_text: per-page text with lang/ocr flags
      - pXX_kv:   optional AI key-value fields per page
      - pXX_tblYY: each extracted table
      - first_schedule_items / second_schedule_items: normalized rows (if present)
      - rules, meta, all_text, all_tables: helpful summary sheets
    """
    import json as _json
    import pandas as _pd

    sheets: dict = {}
    all_text, all_tables = [], []

    for p in doc.pages:
        # per-page text
        sname = f"p{p.page_index:02d}_text"[:31]
        df = _pd.DataFrame({"text": [p.text]})
        df.insert(0, "lang", p.lang)
        df.insert(0, "ocr", p.ocr)
        sheets[sname] = df
        all_text.append({"page": p.page_index, "lang": p.lang, "ocr": p.ocr, "text": p.text})

        # per-page KV (if AI extraction used)
        if p.kv_fields:
            rows = [{"key": k, **fv.model_dump()} for k, fv in p.kv_fields.items()]
            sheets[f"p{p.page_index:02d}_kv"[:31]] = _pd.DataFrame(rows)

        # per-table
        for t in p.tables:
            cols = [c.name for c in t.columns]
            df = _pd.DataFrame(t.rows, columns=cols)
            df.insert(0, "page", p.page_index)
            sheets[t.name[:31]] = df
            for r in t.rows:
                all_tables.append({
                    "page": p.page_index,
                    **{cols[i]: (r[i] if i < len(cols) else None) for i in range(len(cols))}
                })

    # normalized schedules
    if getattr(doc, "normalized", None):
        if "first_schedule_items" in doc.normalized:
            n1 = _pd.DataFrame(doc.normalized["first_schedule_items"])
            if "bullets" in n1.columns:
                n1["bullets_joined"] = n1["bullets"].apply(
                    lambda x: ("\n• " + "\n• ".join(x)) if isinstance(x, list) and x else ""
                )
            sheets["first_schedule_items"] = n1
        if "second_schedule_items" in doc.normalized:
            n2 = _pd.DataFrame(doc.normalized["second_schedule_items"])
            if "bullets" in n2.columns:
                n2["bullets_joined"] = n2["bullets"].apply(
                    lambda x: ("\n• " + "\n• ".join(x)) if isinstance(x, list) and x else ""
                )
            sheets["second_schedule_items"] = n2

    # rules/meta
    if getattr(doc, "rules", None):
        sheets["rules"] = _pd.DataFrame([{"key": k, "value": v} for k, v in doc.rules.items()])
    if getattr(doc, "meta", None):
        meta_rows = []
        for k, v in doc.meta.items():
            if isinstance(v, (list, dict)):
                meta_rows.append({"key": k, "value": _json.dumps(v, ensure_ascii=False)})
            else:
                meta_rows.append({"key": k, "value": v})
        sheets["meta"] = _pd.DataFrame(meta_rows)

    # combined helpers
    if all_text:
        sheets["all_text"] = _pd.DataFrame(all_text)
    if all_tables:
        sheets["all_tables"] = _pd.DataFrame(all_tables)

    return sheets
