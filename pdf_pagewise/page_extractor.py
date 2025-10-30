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

# OpenAI client + typed errors
from openai import OpenAI
from openai import AuthenticationError, APIConnectionError, APIError, RateLimitError, BadRequestError

from schema import (
    CanonicalDoc, DocSource, PageModel, TableModel, TableColumn, FieldValue, OCRLine
)
from postprocess import (
    normalize_first_schedule_items,
    normalize_second_schedule_items,
    clean_description,
)

# ---------- constants ----------
MAX_CHARS_PER_PAGE = 5000
MAX_DOC_CHARS = 15000
DEFAULT_MODEL = "gpt-4o-mini"

_tess_cmd = os.getenv("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# ---------- small error types to signal UI nicely ----------
class LLMAuthError(Exception): ...
class LLMRuntimeError(Exception): ...

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
    """OCR the entire PDF once; return OCR'd PDF bytes, or None if not available."""
    if not _which("ocrmypdf"):
        return None
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "in.pdf"
        out = pathlib.Path(td) / "out.pdf"
        with open(inp, "wb") as f:
            f.write(file_bytes)
        cmd = [
            "ocrmypdf", "--force-ocr", "--jobs", "2", "--output-type", "pdf",
            "--language", langs, "--skip-text", str(inp), str(out)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if out.exists():
                return out.read_bytes()
        except Exception:
            return None
    return None

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
    pad = max(5, int(0.01*min(h, w)))
    inner = bin_img[pad:h-pad, pad:w-pad]
    res = np.pad(inner, ((pad,pad),(pad,pad)), mode="constant", constant_values=255)
    return res

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

# ---------------- OCR passes (Tesseract) ----------------
def _tesseract_pass(pil_img: Image.Image, langs: str, psm: str, oem: str) -> Tuple[str, float]:
    cfg = f"--psm {psm} --oem {oem}"
    np_img = np.array(pil_img)
    data = pytesseract.image_to_data(np_img, lang=langs, config=cfg, output_type=Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if c not in (-1, "-1")]
    txt = pytesseract.image_to_string(np_img, lang=langs, config=cfg) or ""
    conf = (sum(confs)/len(confs)) if confs else -1.0
    return txt, conf

def _tesseract_lines(pil_img: Image.Image, langs: str, psm: str, oem: str) -> List[Dict[str, Any]]:
    """Return list of {'poly':[[x,y]...],'text':str,'conf':float} via image_to_data."""
    cfg = f"--psm {psm} --oem {oem}"
    d = pytesseract.image_to_data(np.array(pil_img), lang=langs, config=cfg, output_type=Output.DICT)
    n = len(d.get("text", []))
    lines: List[Dict[str, Any]] = []
    for i in range(n):
        txt = d["text"][i]
        if not txt or not str(txt).strip():
            continue
        try:
            x, y, w, h = int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i])
            conf_raw = d["conf"][i]
            conf = float(conf_raw) if conf_raw not in (-1, "-1") else -1.0
        except Exception:
            continue
        poly = [[float(x), float(y)], [float(x+w), float(y)], [float(x+w), float(y+h)], [float(x), float(y+h)]]
        lines.append({"poly": poly, "text": str(txt), "conf": conf})
    return lines

def _looks_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (s or ""))

def _ocr_best_of(
    pil_img: Image.Image,
    *,
    speed_mode: bool,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    lang_hint_text: str = ""
) -> str:
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

    for psm in psms:
        try_one(langs_primary, psm)
    if len(best[0].strip()) < 40 and not speed_mode:
        for psm in psms:
            try_one(langs_combo, psm)

    return best[0]

# ---------------- optional PP-OCR-like (EasyOCR) ----------------
def _easyocr_langs_from_env() -> List[str]:
    env = os.getenv("OCR_LANGS", "chi_sim+chi_tra+eng+msa")
    mapping = {"chi_sim": "ch_sim", "chi_tra": "ch_tra", "eng": "en", "msa": "ms", "mal": "ms"}
    out: List[str] = []
    for tok in re.split(r"[+,\s]+", env):
        tok = tok.strip()
        if not tok:
            continue
        out.append(mapping.get(tok, tok))
    if "en" not in out:
        out.append("en")
    # keep order; de-dupe
    seen = set(); res = []
    for a in out:
        if a not in seen:
            seen.add(a); res.append(a)
    return res

def _ocr_ppocr_like(pil_img: Image.Image, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """PP-OCR-like det+rec via easyocr if installed; otherwise return empty (caller will fallback)."""
    try:
        import easyocr  # optional
    except Exception:
        return "", []
    langs = _easyocr_langs_from_env()
    if progress_cb: progress_cb({"stage": "ocr_try", "langs": f"PP-OCR-like:{'+'.join(langs)}", "psm": "det+rec"})
    # GPU False by default; set env EASYOCR_GPU=1 to enable if CUDA available
    reader = easyocr.Reader(langs, gpu=(os.getenv("EASYOCR_GPU","0") == "1"))
    np_img = np.array(pil_img.convert("RGB"))
    res = reader.readtext(np_img, detail=1, paragraph=False)  # [(bbox, text, conf), ...]
    lines, texts = [], []
    for bbox, text, conf in res:
        poly = [[float(x), float(y)] for (x, y) in bbox]
        lines.append({"poly": poly, "text": str(text), "conf": float(conf)})
        if str(text).strip():
            texts.append(str(text).strip())
    return ("\n".join(texts).strip(), lines)

# ---------------- table extraction ----------------
PLUMBER_SETTINGS = [
    {"vertical_strategy": "lines", "horizontal_strategy": "lines",
     "intersection_x_tolerance": 5, "intersection_y_tolerance": 5},
    {"vertical_strategy": "text", "horizontal_strategy": "text",
     "snap_tolerance": 3, "join_tolerance": 3},
]

def _extract_tables_for_page(page: pdfplumber.page.Page, idx1: int) -> List[TableModel]:
    frames: List[TableModel] = []
    seen = set()

    def _nonempty_ratio(rows):
        total = len(rows)
        if not total: return 0.0
        nonempty = sum(1 for r in rows if any((c or "").strip() for c in r))
        return nonempty/total

    def _is_first_schedule_header(header):
        head = " ".join([str(x or "") for x in header]).lower()
        return ("tariff" in head and "code" in head) or ("description" in head)

    tables = []
    for S in PLUMBER_SETTINGS:
        try:
            tbs = page.extract_tables(table_settings=S) or []
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

# ---------------- lang heuristics ----------------
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

# ---------------- OpenAI helpers ----------------
def _llm_extract_kv(text: str, fields: List[str], api_key: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    client = OpenAI(api_key=(api_key or "").strip())
    clipped = (text or "")[:MAX_CHARS_PER_PAGE]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an information extraction engine. Return ONLY valid JSON. If a field is missing, use null."},
                {"role": "user", "content": f"Extract the following fields from this page's text.\nFields: {fields}.\n\n{clipped}"},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except AuthenticationError as e:
        # bubble up a clean signal; UI will show a nice banner
        raise LLMAuthError("OpenAI authentication failed (invalid API key).") from e
    except (RateLimitError, APIConnectionError, APIError, BadRequestError) as e:
        raise LLMRuntimeError(str(e)) from e
    except Exception as e:
        raise LLMRuntimeError(str(e)) from e

def _llm_doc_summary(text: str, api_key: str, model: str = DEFAULT_MODEL) -> dict:
    if not api_key or not (text or "").strip():
        return {}
    client = OpenAI(api_key=(api_key or "").strip())
    clipped = (text or "")[:MAX_DOC_CHARS]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You summarize documents. Return ONLY JSON."},
                {"role": "user", "content": "Summarize the document. Respond as JSON with keys: `summary` (≤200 words) and `key_points` (5–8 bullets).\n\nTEXT:\n" + clipped},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except AuthenticationError as e:
        raise LLMAuthError("OpenAI authentication failed (invalid API key).") from e
    except (RateLimitError, APIConnectionError, APIError, BadRequestError) as e:
        raise LLMRuntimeError(str(e)) from e
    except Exception as e:
        raise LLMRuntimeError(str(e)) from e

# --- single-image entry point (JPG/PNG -> CanonicalDoc) ---
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

    txt = ""
    lines_raw: List[Dict[str, Any]] = []
    ocr_used = True

    if use_ppocr_like:
        txt_pp, lines_raw = _ocr_ppocr_like(pil_bin, progress_cb=progress_cb)
        if txt_pp.strip():
            txt = txt_pp
        else:
            # fallback to Tesseract best-of + line boxes
            if progress_cb: progress_cb({"stage": "ocr", "i": 1, "n": 1})
            txt = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text="")
            oem = os.getenv("OCR_OEM","1")
            lines_raw = _tesseract_lines(pil_bin, os.getenv("OCR_LANGS","chi_sim+chi_tra+eng+msa"), "6", oem)
    else:
        if progress_cb: progress_cb({"stage": "ocr", "i": 1, "n": 1})
        txt = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text="")
        oem = os.getenv("OCR_OEM","1")
        lines_raw = _tesseract_lines(pil_bin, os.getenv("OCR_LANGS","chi_sim+chi_tra+eng+msa"), "6", oem)

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
                for k, v in kv_raw.items()
            }
        except LLMAuthError as e:
            meta_flags["ai_error"] = str(e); ai_ok = False  # stop further AI calls
        except LLMRuntimeError as e:
            meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"

        # nosec

    sha = _sha256_bytes(img_bytes); ts = time.strftime("%Y%m%d_%H%M%S")
    page_lines = [OCRLine(text=l["text"], conf=l.get("conf"), poly=l["poly"]) for l in (lines_raw or [])]
    doc = CanonicalDoc(
        doc_id=f"{ts}_{sha[:8]}",
        source=DocSource(filename=filename, pages=1, sha256=sha),
        pages=[PageModel(page_index=1, text=txt, tables=[], kv_fields=kv_fields or None, lang=lang, ocr=ocr_used, lines=page_lines)],
    )

    if ai_ok and api_key:
        try:
            meta = _llm_doc_summary(txt or "", api_key=api_key, model=model)
            if meta:
                doc.meta = (doc.meta or {}) | {"doc_summary": meta.get("summary"), "doc_key_points": meta.get("key_points", [])}
        except LLMAuthError as e:
            meta_flags["ai_error"] = str(e)
        except LLMRuntimeError as e:
            meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"  # noqa: E501

    if meta_flags:
        doc.meta = (doc.meta or {}) | meta_flags

    if progress_cb: progress_cb({"stage":"page_done", "i":1, "n":1, "sec":0.0})
    return doc

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
    use_ppocr_like: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CanonicalDoc:

    sha = _sha256_bytes(file_bytes)
    ts = time.strftime("%Y%m%d_%H%M%S")
    doc_id = f"{ts}_{sha[:8]}"

    base_pdf = pdfplumber.open(io.BytesIO(file_bytes))

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

    # track AI status once to avoid repeated 401s
    ai_ok = bool(api_key and fields)
    meta_flags: Dict[str, Any] = {}

    for idx1 in range(1, total_pages + 1):
        page_start = time.perf_counter()
        if progress_cb: progress_cb({"stage":"page_start", "i":idx1, "n":total_pages})

        txt = _text_layer(ocr_pdf_obj or base_pdf, idx1)
        if len(txt.strip()) < 20:
            if progress_cb: progress_cb({"stage":"pdftotext", "i":idx1, "n":total_pages})
            txt = _pdftotext_page(file_bytes, idx1) or txt

        ocr_used = False
        page_lines: List[OCRLine] = []

        if len(txt.strip()) < 60:
            dpi = int(os.getenv("OCR_DPI", "300" if speed_mode else "400"))
            imgs = convert_from_bytes(file_bytes, dpi=dpi, first_page=idx1, last_page=idx1)
            if imgs:
                if progress_cb: progress_cb({"stage":"preprocess", "i":idx1, "n":total_pages})
                pil = imgs[0]
                pil_bin = _opencv_preprocess(pil) if aggressive_ocr else pil

                # PP-OCR-like first if requested
                if use_ppocr_like:
                    txt_pp, lines_raw = _ocr_ppocr_like(pil_bin, progress_cb=progress_cb)
                    if txt_pp.strip():
                        txt = txt_pp
                        page_lines = [OCRLine(text=l["text"], conf=l.get("conf"), poly=l["poly"]) for l in lines_raw]
                        ocr_used = True
                    else:
                        # fallback to Tesseract best-of + line boxes
                        if progress_cb: progress_cb({"stage":"ocr", "i":idx1, "n":total_pages})
                        txt2 = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text=txt)
                        oem = os.getenv("OCR_OEM","1")
                        lines_raw = _tesseract_lines(pil_bin, os.getenv("OCR_LANGS","chi_sim+chi_tra+eng+msa"), "6", oem)
                        if len(txt2.strip()) > len(txt.strip()):
                            txt = txt2; ocr_used = True
                        page_lines = [OCRLine(text=l["text"], conf=l.get("conf"), poly=l["poly"]) for l in lines_raw]
                else:
                    if progress_cb: progress_cb({"stage":"ocr", "i":idx1, "n":total_pages})
                    txt2 = _ocr_best_of(pil_bin, speed_mode=speed_mode, progress_cb=progress_cb, lang_hint_text=txt)
                    oem = os.getenv("OCR_OEM","1")
                    lines_raw = _tesseract_lines(pil_bin, os.getenv("OCR_LANGS","chi_sim+chi_tra+eng+msa"), "6", oem)
                    if len(txt2.strip()) > len(txt.strip()):
                        txt = txt2; ocr_used = True
                    page_lines = [OCRLine(text=l["text"], conf=l.get("conf"), poly=l["poly"]) for l in lines_raw]

        if progress_cb: progress_cb({"stage":"tables", "i":idx1, "n":total_pages})
        tables = _extract_tables_for_page((ocr_pdf_obj or base_pdf).pages[idx1 - 1], idx1)

        kv_fields: Dict[str, FieldValue] = {}
        if ai_ok:
            try:
                if progress_cb: progress_cb({"stage":"ai", "i":idx1, "n":total_pages})
                kv_raw = _llm_extract_kv(txt, fields or [], api_key, model=model)
                for k, v in kv_raw.items():
                    kv_fields[k] = FieldValue(value=v, type=("number" if isinstance(v, (int,float)) else "string"), page=idx1)
            except LLMAuthError as e:
                meta_flags["ai_error"] = str(e); ai_ok = False  # stop further AI calls
            except LLMRuntimeError as e:
                meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"

        pages.append(PageModel(
            page_index=idx1,
            text=(txt[:MAX_CHARS_PER_PAGE] if txt else ""),
            tables=tables,
            kv_fields=kv_fields or None,
            lang=_guess_lang_for_page(txt),
            ocr=ocr_used,
            lines=page_lines or []
        ))

        if progress_cb:
            progress_cb({"stage":"page_done", "i":idx1, "n":total_pages, "sec": time.perf_counter()-page_start})

    try: base_pdf.close()
    except Exception: pass
    try:
        if ocr_pdf_obj: ocr_pdf_obj.close()
    except Exception: pass

    source = DocSource(filename=filename, pages=len(pages), sha256=sha, lang=lang)
    doc = CanonicalDoc(doc_id=doc_id, source=source, pages=pages)

    # meta baseline
    doc.meta = {
        "cover_text": (pages[0].text[:600] if pages else None),
        "language_pages": [{"page_index": p.page_index, "lang": p.lang} for p in pages if p.lang],
    }

    full_text = "\n".join(p.text for p in pages if p.text)
    if ai_ok and api_key:
        try:
            _meta = _llm_doc_summary(full_text, api_key=api_key, model=model)
            if _meta:
                doc.meta = (doc.meta or {}) | {"doc_summary": _meta.get("summary"), "doc_key_points": _meta.get("key_points", [])}
        except LLMAuthError as e:
            meta_flags["ai_error"] = str(e); ai_ok = False
        except LLMRuntimeError as e:
            meta_flags["ai_warning"] = f"OpenAI error: {e.__class__.__name__}: {e}"

    if "first schedule" in full_text.lower():
        doc.doc_type = "sales_tax_schedule"
    rate = _extract_first_schedule_rate_from_text(full_text)
    if rate:
        doc.rules = (doc.rules or {}); doc.rules["first_schedule_rate"] = rate

    first_items = normalize_first_schedule_items(pages, rate, doc_id)
    if first_items:
        doc.normalized = (doc.normalized or {}) | {"first_schedule_items": first_items}
    second_items = normalize_second_schedule_items(pages, doc_id)
    if second_items:
        doc.normalized = (doc.normalized or {}) | {"second_schedule_items": second_items}

    if meta_flags:
        doc.meta = (doc.meta or {}) | meta_flags

    return doc

# ---------------- pdftotext fallback ----------------
def _pdftotext_page(file_bytes: bytes, page_num: int) -> str:
    if not _which("pdftotext"):
        return ""
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "in.pdf"
        out = pathlib.Path(td) / "out.txt"
        with open(inp, "wb") as f:
            f.write(file_bytes)
        try:
            subprocess.run(["pdftotext", "-f", str(page_num), "-l", str(page_num), str(inp), str(out)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if out.exists():
                return out.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""

# ---------------- Excel packer ----------------
def build_excel_sheets(doc: CanonicalDoc) -> Dict[str, "pd.DataFrame"]:
    import pandas as _pd
    import json as _json

    sheets: dict = {}
    all_text, all_tables = [], []

    for p in doc.pages:
        sname = f"p{p.page_index:02d}_text"[:31]
        df = _pd.DataFrame({"text": [p.text]})
        df.insert(0, "lang", p.lang)
        df.insert(0, "ocr", p.ocr)
        sheets[sname] = df
        all_text.append({"page": p.page_index, "lang": p.lang, "ocr": p.ocr, "text": p.text})

        # NEW: per-line detections
        if getattr(p, "lines", None):
            # flatten polygon to x1..x4,y1..y4 for Excel friendliness
            rows = []
            for L in p.lines:
                poly = L.poly or [[None,None]]*4
                # pad/trim to 4 points
                poly = (poly + [[None,None]]*4)[:4]
                row = {
                    "page": p.page_index,
                    "x1": poly[0][0], "y1": poly[0][1],
                    "x2": poly[1][0], "y2": poly[1][1],
                    "x3": poly[2][0], "y3": poly[2][1],
                    "x4": poly[3][0], "y4": poly[3][1],
                    "text": L.text,
                    "conf": L.conf,
                }
                rows.append(row)
            sheets[f"p{p.page_index:02d}_lines"[:31]] = _pd.DataFrame(rows)

        if p.kv_fields:
            rows = [{"key": k, **fv.model_dump()} for k, fv in p.kv_fields.items()]
            sheets[f"p{p.page_index:02d}_kv"[:31]] = _pd.DataFrame(rows)

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

    if getattr(doc, "rules", None):
        rrows = [{"key": k, "value": v} for k, v in doc.rules.items()]
        sheets["rules"] = _pd.DataFrame(rrows)
    if getattr(doc, "meta", None):
        meta_rows = []
        for k, v in doc.meta.items():
            if isinstance(v, (list, dict)):
                meta_rows.append({"key": k, "value": _json.dumps(v, ensure_ascii=False)})
            else:
                meta_rows.append({"key": k, "value": v})
        sheets["meta"] = _pd.DataFrame(meta_rows)

    if all_text:
        sheets["all_text"] = _pd.DataFrame(all_text)
    if all_tables:
        sheets["all_tables"] = _pd.DataFrame(all_tables)

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
            sheets["second_schedule_items"] = n2

    return sheets
