import io
import os
import json
import time
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from page_extractor import extract_pagewise, extract_image, build_excel_sheets, DEFAULT_MODEL

# ----------------- env -----------------
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ----------------- UI helpers -----------------
def _progress_renderer():
    stage_line = st.empty()
    bar = st.progress(0.0)
    page_times: List[float] = []
    state = {"done_pages": 0, "total_pages_est": 0}

    def cb(ev: Dict[str, Any]):
        stage = ev.get("stage")
        i, n = int(ev.get("i", 0)), int(ev.get("n", 0))
        if n:
            state["total_pages_est"] = n
        labels = {
            "page_start": "Starting page",
            "ocrmypdf": "OCRmyPDF (whole file prepass)",
            "pdftotext": "Extracting text layer",
            "preprocess": "Preprocessing (denoise/deskew/threshold)",
            "ocr": "OCR (best-of passes)",
            "ocr_try": f"OCR try (langs={ev.get('langs')}, psm={ev.get('psm')})",
            "tables": "Extracting tables",
            "ai": "Extracting AI fields",
            "page_done": "Finishing page",
        }
        if stage in labels:
            info = labels[stage]
            if i and n:
                info = f"{info} — page {i}/{n}"
            stage_line.write(f"**Stage:** {info}")

        if stage == "page_done":
            sec = float(ev.get("sec", 0.0))
            if sec > 0:
                page_times.append(sec)
            state["done_pages"] += 1

        # progress bar
        if state["total_pages_est"]:
            frac = state["done_pages"] / max(1, state["total_pages_est"])
            bar.progress(min(1.0, frac))

    return cb

def _download_buttons(doc_json: dict, excel_sheets: Dict[str, pd.DataFrame]):
    # JSON
    json_bytes = json.dumps(doc_json, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ Download JSON", data=json_bytes, file_name="doc.json", mime="application/json")

    # Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        for name, df in excel_sheets.items():
            # Excel sheet name limit 31
            df.to_excel(w, index=False, sheet_name=name[:31])
    st.download_button("⬇️ Download Excel", data=out.getvalue(), file_name="doc.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------- App -----------------
st.set_page_config(page_title="PDF → AI-Readable (PP-OCR-like)", layout="wide")
st.title("📄 PDF → CSV/JSON (with optional PP-OCR-like det+rec)")

with st.sidebar:
    st.header("Settings")

    aggressive_ocr = st.checkbox(
        "Aggressive preprocess (denoise/deskew)",
        value=True,
        help="OpenCV denoise, adaptive threshold, deskew."
    )
    try_ocrmypdf = st.checkbox(
        "Try OCRmyPDF pre-pass",
        value=True,
        help="If the text layer is weak, run OCRmyPDF once for the whole file."
    )
    speed_mode = st.checkbox(
        "Speed mode (fast)",
        value=True,
        help="Use fewer Tesseract passes."
    )
    # NEW: PP-OCR-like toggle
    use_ppocr_like = st.checkbox(
        "Experimental: PP-OCR-like (det+rec lines)",
        value=False,
        help="Uses easyocr if installed; otherwise falls back to Tesseract line boxes."
    )

    st.divider()
    call_ai = st.checkbox("Call OpenAI for fields/summary", value=False)
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    model = st.text_input("Model", value=DEFAULT_MODEL)

    fields_raw = st.text_area(
        "Fields to extract (comma-separated)",
        value="HS Code, Description, Rate",
        help="Used only if Call OpenAI is enabled."
    )
    fields = [f.strip() for f in fields_raw.split(",") if f.strip()] if call_ai else None

st.write("Upload a **PDF** or an **image** (PNG/JPG).")

up = st.file_uploader("Choose file", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)

if up:
    fb = up.read()
    ext = (up.name.split(".")[-1] or "").lower()
    cb = _progress_renderer()

    try:
        if ext == "pdf":
            doc = extract_pagewise(
                fb, up.name,
                api_key=(api_key if call_ai else ""),
                model=model,
                fields=fields,
                aggressive_ocr=aggressive_ocr,
                try_ocrmypdf=try_ocrmypdf,
                speed_mode=speed_mode,
                progress_cb=cb,
                use_ppocr_like=use_ppocr_like,   # NEW
            )
        else:
            doc = extract_image(
                fb, up.name,
                api_key=(api_key if call_ai else ""),
                model=model,
                fields=fields,
                aggressive_ocr=aggressive_ocr,
                speed_mode=speed_mode,
                progress_cb=cb,
                use_ppocr_like=use_ppocr_like,   # NEW
            )
    except Exception as e:
        st.error(f"Extraction error: {e}")
        st.stop()

    # ---- Results ----
    st.success("Done!")

    # per-page preview
    for p in doc.pages:
        with st.expander(f"Page {p.page_index}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area("Text", p.text, height=220)
            with col2:
                st.write("Meta")
                st.json({"lang": p.lang, "ocr_used": p.ocr})

            # lines table
            if getattr(p, "lines", None):
                line_rows = []
                for L in p.lines:
                    poly = L.poly if isinstance(L.poly, list) else []
                    poly = (poly + [[None, None]] * 4)[:4]
                    line_rows.append({
                        "x1": poly[0][0], "y1": poly[0][1],
                        "x2": poly[1][0], "y2": poly[1][1],
                        "x3": poly[2][0], "y3": poly[2][1],
                        "x4": poly[3][0], "y4": poly[3][1],
                        "text": L.text, "conf": L.conf,
                    })
                st.caption("Detected lines")
                st.dataframe(pd.DataFrame(line_rows), use_container_width=True)

            # kv fields (AI)
            if p.kv_fields:
                rows = [{"key": k, **v.model_dump()} for k, v in p.kv_fields.items()]
                st.caption("AI Key/Value Fields")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # tables
            if p.tables:
                st.caption("Tables")
                for t in p.tables:
                    df = pd.DataFrame(t.rows, columns=[c.name for c in t.columns])
                    st.write(f"**{t.name}**")
                    st.dataframe(df, use_container_width=True)

    # doc meta/normalized
    with st.expander("Document Meta & Normalized Outputs", expanded=False):
        st.json({
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "rules": doc.rules,
            "meta": doc.meta,
            "normalized_keys": list((doc.normalized or {}).keys())
        })

    # downloads
    sheets = build_excel_sheets(doc)
    _download_buttons(json.loads(doc.model_dump_json()), sheets)

# friendly footer
st.caption("Tip: Set env `OCR_LANGS=chi_sim+chi_tra+eng+msa` and `EASYOCR_GPU=1` if you have a CUDA GPU.")
