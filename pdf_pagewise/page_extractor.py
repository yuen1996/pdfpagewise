from __future__ import annotations

import io
import os
import re
import cv2
import time
import math
import hashlib
import sys
import logging
import tempfile
import subprocess
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Callable, Tuple

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore

# Optional and best-effort
try:
    import easyocr  # PP-OCR like (det + rec)
except Exception:
    easyocr = None  # type: ignore

# Optional pdf handling
try:
    import pdfplumber
except Exception:
    pdfplumber = None  # type: ignore

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None  # type: ignore

from schema import (
    CanonicalDoc, DocSource, PageModel, TableModel, TableColumn, FieldValue, OCRLine
)

# ------------------------- Globals -------------------------

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OCR_DEFAULT_LANGS = os.getenv("OCR_LANGS", "eng+msa")  # English + Malay
OCR_DEFAULT_PSM = os.getenv("OCR_PSM", "6")
OCR_OEM = os.getenv("OCR_OEM", "1")
OCR_DPI = int(os.getenv("OCR_DPI", "220"))  # tunable PDF raster DPI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------- Utility helpers -------------------------

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _remove_outer_border(img_bin: np.ndarray) -> np.ndarray:
    """Crop thin solid borders that confuse OCR on posters."""
    h, w = img_bin.shape[:2]
    margin = int(max(h, w) * 0.02)
    if margin <= 0:
        return img_bin
    inner = img_bin[margin:h - margin, margin:w - margin]
    return inner if inner.size else img_bin


def _deskew_by_hough(img_bin: np.ndarray) -> np.ndarray:
    """Very light Hough-based deskew for binary images."""
    try:
        edges = cv2.Canny(img_bin, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
        if lines is None:
            return img_bin
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * (180.0 / np.pi)
            if -30 < angle < 30:
                angles.append(angle)
        if not angles:
            return img_bin
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.1:
            return img_bin
        (h, w) = img_bin.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img_bin, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img_bin


def _opencv_preprocess(pil_img: Image.Image) -> Image.Image:
    """Contrast/denoise/threshold for colored posters, then deskew & crop borders."""
    img = np.array(pil_img.convert("RGB"))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    norm = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    thr = cv2.medianBlur(thr, 3)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    thr = _remove_outer_border(thr)
    thr = _deskew_by_hough(thr)
    return Image.fromarray(thr)


# ------------------------- OCR engines -------------------------

def _tesseract_lines(pil_or_cv_img: Image.Image, langs: str, psm: str, oem: str) -> List[Dict[str, Any]]:
    """Return line boxes using pytesseract image_to_data."""
    if pytesseract is None:
        return []
    if isinstance(pil_or_cv_img, Image.Image):
        pil = pil_or_cv_img
    else:
        pil = Image.fromarray(pil_or_cv_img)

    cfg = f'--psm {psm} --oem {oem}'
    data = pytesseract.image_to_data(pil, lang=langs, config=cfg, output_type=pytesseract.Output.DICT)
    n = len(data["text"])
    lines: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for i in range(n):
        if not data["text"][i] or str(data["conf"][i]).strip() == "" or int(float(data["conf"][i])) < 0:
            continue
        page_num = data.get("page_num", [1] * n)[i]
        line_num = data.get("line_num", [i] * n)[i]
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        key = (page_num, line_num)
        entry = lines.get(key)
        if entry is None:
            entry = {"text": [], "conf": [], "x": x, "y": y, "x2": x + w, "y2": y + h}
        entry["text"].append(str(data["text"][i]))
        try:
            entry["conf"].append(float(data["conf"][i]))
        except Exception:
            pass
        entry["x"] = min(entry["x"], x); entry["y"] = min(entry["y"], y)
        entry["x2"] = max(entry["x2"], x + w); entry["y2"] = max(entry["y2"], y + h)
        lines[key] = entry

    out: List[Dict[str, Any]] = []
    for _, L in sorted(lines.items(), key=lambda kv: (kv[1]["y"], kv[1]["x"])):
        text = " ".join([t for t in L["text"] if t is not None]).strip()
        confs = [c for c in L["conf"] if c is not None]
        conf = float(np.mean(confs)) if confs else None
        x, y, w, h = L["x"], L["y"], L["x2"] - L["x"], L["y2"] - L["y"]
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        out.append({"text": text, "conf": conf, "poly": poly})
    return out


def _ocr_best_of(pil_img: Image.Image, speed_mode: bool = True,
                 progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                 lang_hint_text: str = "") -> str:
    """Try a couple of Tesseract settings and return the best."""
    if pytesseract is None:
        return ""
    langs = OCR_DEFAULT_LANGS
    psms = ["6", "4"] if speed_mode else ["6", "4", "3"]
    best_text = ""
    for psm in psms:
        if progress_cb:
            progress_cb({"stage": "ocr_try", "langs": langs, "psm": psm})
        cfg = f'--psm {psm} --oem {OCR_OEM}'
        try:
            text = pytesseract.image_to_string(pil_img, lang=langs, config=cfg)
        except Exception:
            continue
        if len(text.strip()) > len(best_text.strip()):
            best_text = text
    return best_text


def _ocr_ppocr_like(pil_img: Image.Image, *,
                    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Use EasyOCR (det+rec) to get line boxes similar to PP-OCR."""
    if easyocr is None:
        return "", []
    try:
        gpu_flag = bool(int(os.getenv("EASYOCR_FORCE_GPU", "0")))
    except Exception:
        gpu_flag = False
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_flag = True
    except Exception:
        pass

    langs = list(set(OCR_DEFAULT_LANGS.replace("+", " ").replace(",", " ").split()))
    reader = easyocr.Reader(langs, gpu=gpu_flag, verbose=False)
    if progress_cb:
        progress_cb({"stage": "ocr", "engine": "easyocr", "gpu": gpu_flag})
    res = reader.readtext(np.array(pil_img))
    lines: List[Dict[str, Any]] = []
    texts: List[str] = []
    for box, text, conf in res:
        try:
            poly = [[float(x), float(y)] for x, y in box]
        except Exception:
            x1 = min([p[0] for p in box]); y1 = min([p[1] for p in box])
            x2 = max([p[0] for p in box]); y2 = max([p[1] for p in box])
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        text = (text or "").strip()
        if text:
            lines.append({"text": text, "conf": float(conf), "poly": poly})
            texts.append(text)
    joined = "\n".join(texts)
    return joined, lines


# ------------------------- Reading-order rebuild -------------------------

def _compose_text_from_lines(lines: List[Dict[str, Any]]) -> str:
    """Rebuild readable text by sorting lines and grouping rows."""
    if not lines:
        return ""
    items = []
    for L in lines:
        poly = L.get("poly") or []
        if not poly:
            continue
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x_left = float(min(xs)); y_base = float(sum(ys) / len(ys))
        height = float(max(ys) - min(ys)) if ys else 12.0
        txt = str(L.get("text") or "").strip()
        if txt:
            items.append((y_base, x_left, height, txt))
    if not items:
        return ""
    items.sort(key=lambda t: (t[0], t[1]))
    import statistics as _stats
    try: med_h = _stats.median([h for _, _, h, _ in items])
    except Exception: med_h = 14.0
    row_tol = max(8.0, 0.6 * med_h)
    rows = []; cur_y = None; cur_row = []
    for y, x, h, txt in items:
        if cur_y is None or abs(y - cur_y) <= row_tol:
            cur_row.append((x, txt)); cur_y = y if cur_y is None else (cur_y + y) / 2.0
        else:
            cur_row.sort(key=lambda t: t[0]); rows.append(cur_row)
            cur_row = [(x, txt)]; cur_y = y
    if cur_row:
        cur_row.sort(key=lambda t: t[0]); rows.append(cur_row)
    lines_out = [" ".join(t for _, t in r) for r in rows]
    return "\n".join(lines_out).strip()


# ------------------------- PDF helpers (OCRmyPDF + text layer) -------------------------

def _run_ocrmypdf(pdf_bytes: bytes) -> bytes:
    """Try to run OCRmyPDF; return improved PDF bytes or original on failure."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as inf:
            inf.write(pdf_bytes); in_path = inf.name
        out_path = in_path + ".ocr.pdf"
        jobs = str(max(2, (os.cpu_count() or 2)))
        cmd = [
            "ocrmypdf",
            "--skip-text",          # only OCR pages without a text layer
            "--deskew",
            "--rotate-pages",
            "--jobs", jobs,
            "--optimize", "0",
            in_path, out_path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                return f.read()
        else:
            logger.warning("ocrmypdf failed rc=%s: %s", proc.returncode, proc.stderr.decode(errors="ignore")[:400])
            return pdf_bytes
    except Exception as e:
        logger.warning("ocrmypdf exception: %s", e)
        return pdf_bytes
    finally:
        try: os.remove(in_path)
        except Exception: pass
        try: os.remove(out_path)
        except Exception: pass


def _pdf_text_per_page(pdf_bytes: bytes) -> List[Optional[str]]:
    """Return text layer per page using pdfplumber, or [] if unavailable."""
    out: List[Optional[str]] = []
    if pdfplumber is None:
        return out
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for p in pdf.pages:
                try:
                    txt = p.extract_text() or ""
                except Exception:
                    txt = ""
                out.append(txt.strip() or None)
    except Exception as e:
        logger.warning("pdfplumber open failed: %s", e)
    return out


def _pdf_to_images(pdf_bytes: bytes, dpi: int = None) -> List[Image.Image]:
    if convert_from_bytes is None:
        raise RuntimeError("pdf2image not installed")
    _dpi = dpi if dpi is not None else OCR_DPI
    return convert_from_bytes(pdf_bytes, dpi=_dpi)


# ------------------------- Language, AI helpers (safe stubs) -------------------------

def _guess_lang_for_page(text: str) -> Optional[str]:
    txt = text or ""
    if re.search(r"[اأإآء-ي]", txt):
        return "ara"
    if re.search(r"[\u4E00-\u9FFF]", txt):
        return "zho"
    if re.search(r"[A-Za-z]", txt) and re.search(r"[àâäáãåčćđéèëêíìïîñóòöôõřšúùüûž]", txt):
        return "msa"
    if re.search(r"[A-Za-z]", txt):
        return "eng"
    return None


class LLMAuthError(RuntimeError):
    pass


class LLMRuntimeError(RuntimeError):
    pass


def _llm_extract_kv(text: str, fields: List[str], api_key: str, *, model: str) -> Dict[str, Any]:
    if not api_key:
        return {}
    try:
        from openai import OpenAI
    except Exception:
        return {}
    try:
        client = OpenAI(api_key=api_key)
        sys_prompt = (
            "Return a strict JSON object with keys exactly matching this list. "
            "Use null when not present.\nKeys: " + ", ".join(fields)
        )
        user_prompt = text[:6000]
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        try:
            import json as _json
            parsed = _json.loads(raw)
        except Exception:
            parsed = {}
        return {k: parsed.get(k) for k in fields}
    except Exception as e:
        raise LLMRuntimeError(str(e))


def _llm_doc_summary(text: str, *, api_key: str, model: str) -> Dict[str, Any]:
    if not api_key:
        return {}
    try:
        from openai import OpenAI
    except Exception:
        return {}
    try:
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": "Summarize the document briefly and bullet key points."},
            {"role": "user", "content": text[:8000]},
        ]
        completion = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
        summary = (completion.choices[0].message.content or "").strip()
        return {"summary": summary, "key_points": []}
    except Exception as e:
        raise LLMRuntimeError(str(e))


# ------------------------- Core extractors -------------------------

def extract_image(
    img_bytes: bytes,
    filename: str,
    *,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    fields: Optional[List[str]] = None,
    aggressive_ocr: bool = True,
    speed_mode: bool = True,
    use_ppocr_like: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CanonicalDoc:
    if progress_cb: progress_cb({"stage": "page_start", "i": 1, "n": 1})

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil_bin = _opencv_preprocess(pil) if aggressive_ocr else pil

    txt_raw = ""
    lines_raw: List[Dict[str, Any]] = []
    ocr_used = True

    if use_ppocr_like:
        txt_pp, lines_raw = _ocr_ppocr_like(pil_bin, progress_cb=progress_cb)
        if txt_pp.strip():
            txt_raw = txt_pp
        else:
            if progress_cb: progress_cb({"stage": "ocr", "i": 1, "n": 1})
            txt_raw = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text="")
            oem = os.getenv("OCR_OEM", "1")
            lines_raw = _tesseract_lines(pil_bin, os.getenv("OCR_LANGS", OCR_DEFAULT_LANGS), "6", oem)
    else:
        if progress_cb: progress_cb({"stage": "ocr", "i": 1, "n": 1})
        txt_raw = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text="")
        oem = os.getenv("OCR_OEM", "1")
        lines_raw = _tesseract_lines(pil_bin, os.getenv("OCR_LANGS", OCR_DEFAULT_LANGS), "6", oem)

    txt_lines = _compose_text_from_lines(lines_raw)
    def _score(s: str) -> tuple[int, int]:
        import re as _re
        s2 = s.strip()
        return (len(s2), len(_re.findall(r"[A-Za-z0-9%]", s2)))
    txt = txt_lines if _score(txt_lines) >= _score(txt_raw) else txt_raw

    lang = _guess_lang_for_page(txt)

    kv_fields: Dict[str, FieldValue] = {}
    meta_flags: Dict[str, Any] = {}
    ai_ok = bool(api_key and fields)

    if ai_ok:
        try:
            if progress_cb: progress_cb({"stage": "ai", "i": 1, "n": 1})
            kv_raw = _llm_extract_kv(txt, fields or [], api_key, model=model)
            kv_fields = {
                k: FieldValue(value=v, type=("number" if isinstance(v, (int, float)) else "string"), page=1)
                for k, v in (kv_raw or {}).items()
            }
        except LLMAuthError as e:
            meta_flags["ai_error"] = str(e)
            ai_ok = False
        except LLMRuntimeError as e:
            meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"

    sha = _sha256_bytes(img_bytes); ts = time.strftime("%Y%m%d_%H%M%S")
    page_lines = [OCRLine(text=l["text"], conf=l.get("conf"), poly=l["poly"]) for l in (lines_raw or [])]
    doc = CanonicalDoc(
        doc_id=f"{ts}_{sha[:8]}",
        source=DocSource(filename=filename, pages=1, sha256=sha),
        pages=[PageModel(page_index=1, text=txt, tables=[], kv_fields=kv_fields or {}, lang=lang, ocr=ocr_used, lines=page_lines)],
    )

    if ai_ok and api_key:
        try:
            meta = _llm_doc_summary(txt or "", api_key=api_key, model=model)
            if meta:
                doc.meta = (doc.meta or {}) | {"doc_summary": meta.get("summary"), "doc_key_points": meta.get("key_points", [])}
        except LLMAuthError as e:
            meta_flags["ai_error"] = str(e)
        except LLMRuntimeError as e:
            meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"

    if meta_flags:
        doc.meta = (doc.meta or {}) | meta_flags

    if progress_cb: progress_cb({"stage": "page_done", "i": 1, "n": 1, "sec": 0.0})
    return doc


def extract_pagewise(
    file_bytes: bytes,
    filename: str,
    *,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    fields: Optional[List[str]] = None,
    aggressive_ocr: bool = True,
    try_ocrmypdf: bool = False,      # <<< NEW
    speed_mode: bool = True,
    use_ppocr_like: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CanonicalDoc:
    """
    Extract a whole file (PDF or image). For PDFs, process page by page.
    """
    name_lower = (filename or "").lower()
    if name_lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".gif")):
        return extract_image(
            file_bytes, filename, api_key=api_key, model=model, fields=fields,
            aggressive_ocr=aggressive_ocr, speed_mode=speed_mode, use_ppocr_like=use_ppocr_like,
            progress_cb=progress_cb
        )

    # --- PDF path ---
    if progress_cb: progress_cb({"stage": "start_pdf"})
    pdf_bytes = file_bytes

    # Optional OCRmyPDF pre-pass to add/repair text layer
    if try_ocrmypdf:
        if progress_cb: progress_cb({"stage": "ocrmypdf"})
        pdf_bytes = _run_ocrmypdf(pdf_bytes)

    # Get text layer (if any) and images
    text_layer = _pdf_text_per_page(pdf_bytes) if pdfplumber else []
    images = _pdf_to_images(pdf_bytes, dpi=OCR_DPI)
    n = len(images)
    pages: List[PageModel] = []

    sha = _sha256_bytes(file_bytes)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ocr_used = True

    def _score(s: str) -> tuple[int, int]:
        import re as _re
        s2 = (s or "").strip()
        return (len(s2), len(_re.findall(r"[A-Za-z0-9%]", s2)))

    for i, pil in enumerate(images, start=1):
        if progress_cb: progress_cb({"stage": "page_start", "i": i, "n": n})
        pil_bin = _opencv_preprocess(pil) if aggressive_ocr else pil

        # candidates: 1) text layer, 2) best-of Tesseract, 3) EasyOCR (if enabled), 4) reading-order from lines
        txt_candidates: List[str] = []
        tl = (text_layer[i-1] if i-1 < len(text_layer) else None)
        if tl: txt_candidates.append(tl)

        lines_raw: List[Dict[str, Any]] = []
        txt_raw = ""
        if use_ppocr_like:
            txt_pp, lines_raw = _ocr_ppocr_like(pil_bin, progress_cb=progress_cb)
            if txt_pp.strip():
                txt_raw = txt_pp
        if not txt_raw:
            if progress_cb: progress_cb({"stage": "ocr", "i": i, "n": n})
            txt_raw = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb)
            lines_raw = lines_raw or _tesseract_lines(pil_bin, OCR_DEFAULT_LANGS, OCR_DEFAULT_PSM, OCR_OEM)

        txt_candidates.append(txt_raw)
        txt_lines = _compose_text_from_lines(lines_raw)
        txt_candidates.append(txt_lines)

        # choose best by length & alnum density
        best_txt = max(txt_candidates, key=_score, default="")
        lang = _guess_lang_for_page(best_txt)

        page_lines = [OCRLine(text=l["text"], conf=l.get("conf"), poly=l["poly"]) for l in (lines_raw or [])]
        pages.append(PageModel(page_index=i, text=best_txt, tables=[], kv_fields={}, lang=lang, ocr=ocr_used, lines=page_lines))
        if progress_cb: progress_cb({"stage": "page_done", "i": i, "n": n, "sec": 0.0})

    # One-shot AI KV over all text (optional)
    kv_fields: Dict[str, FieldValue] = {}
    all_text = "\n\n".join([p.text for p in pages])
    meta_flags: Dict[str, Any] = {}
    if api_key and fields:
        try:
            if progress_cb: progress_cb({"stage": "ai"})
            kv_raw = _llm_extract_kv(all_text, fields or [], api_key, model=model)
            kv_fields = {
                k: FieldValue(value=v, type=("number" if isinstance(v, (int, float)) else "string"), page=1)
                for k, v in (kv_raw or {}).items()
            }
            if pages:
                pages[0].kv_fields = kv_fields or {}
        except LLMRuntimeError as e:
            meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"
        except LLMAuthError as e:
            meta_flags["ai_error"] = str(e)

    doc = CanonicalDoc(
        doc_id=f"{ts}_{sha[:8]}",
        source=DocSource(filename=filename, pages=len(pages), sha256=sha),
        pages=pages,
    )
    if meta_flags:
        doc.meta = (doc.meta or {}) | meta_flags
    return doc


# ------------------------- Excel writer -------------------------

def build_excel_sheets(doc: CanonicalDoc) -> Dict[str, "pd.DataFrame"]:
    import pandas as pd
    page_rows = [{"page": p.page_index, "lang": p.lang, "ocr": p.ocr, "text": p.text} for p in doc.pages]
    sheets: Dict[str, pd.DataFrame] = {"Pages": pd.DataFrame(page_rows)}

    kv_rows = []
    for p in doc.pages:
        for k, fv in (p.kv_fields or {}).items():
            kv_rows.append({
                "page": p.page_index,
                "key": k,
                "value": fv.value,
                "type": fv.type,
                "unit": fv.unit,
                "conf": fv.confidence
            })
    if kv_rows:
        sheets["KV Fields"] = pd.DataFrame(kv_rows)

    for p in doc.pages:
        for t in p.tables or []:
            cols = [c.name for c in t.columns]
            df = pd.DataFrame(t.rows, columns=cols)
            sheets[f"Tables - Page {p.page_index} - {t.name}"] = df

    return sheets
