import io
import os
import json
import time
from typing import List
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from page_extractor import extract_pagewise, build_excel_sheets, DEFAULT_MODEL

# Load .env from current dir, repo root, and app dir
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

st.set_page_config(page_title="Page-by-Page PDF OCR → JSON → Excel", layout="wide")
st.title("Page-by-Page PDF OCR → JSON → Excel")

with st.sidebar:
    st.header("Settings")

    # OCR controls
    ocr_langs_default = os.getenv("OCR_LANGS", "chi_sim+chi_tra+eng+msa")
    ocr_langs = st.text_input("OCR languages (tesseract codes with +)", value=ocr_langs_default)
    dpi = st.slider("OCR DPI", min_value=200, max_value=600, value=int(os.getenv("OCR_DPI", "400")), step=50)
    aggressive_ocr = st.checkbox("Aggressive OCR (OpenCV + multi-pass)", value=True)
    try_ocrmypdf = st.checkbox("Try OCRmyPDF first (if installed)", value=True)

    # AI extraction
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.selectbox("Model", [DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini"], index=0)
    use_ai = st.checkbox("Use AI extraction (per page)", value=bool(api_key or os.getenv("OPENAI_API_KEY")))
    fields_default = os.getenv("DEFAULT_AI_FIELDS", "page_summary,key_points")
    fields_csv = st.text_input("Fields to extract (comma-separated)", value=fields_default)

    st.caption(f"Langs={ocr_langs}  DPI={dpi}  OEM={os.getenv('OCR_OEM','1')}  PSMs={os.getenv('OCR_PSMS','6,4,11')}")

files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if not files:
    st.info("Upload PDFs to begin.")
    st.stop()

if st.button("Process (Page-by-Page)", type="primary"):
    # set runtime env for extractor
    os.environ["OCR_LANGS"] = ocr_langs
    os.environ["OCR_DPI"] = str(dpi)

    for up in files:
        st.subheader(f"📄 {up.name}")
        fb = up.read()

        # progress UI
        bar = st.progress(0)
        stage_line = st.empty()
        eta_line = st.empty()
        log = st.container()

        start_all = time.perf_counter()
        page_times: List[float] = []

        # mutable state for callback (avoids 'nonlocal' error)
        state = {
            "done_pages": 0,
            "total_pages_est": None,
        }

        # decide AI fields
        call_ai = use_ai and bool(api_key)
        fields_csv_final = (fields_csv or "").strip()
        if call_ai and not fields_csv_final:
            fields_csv_final = "page_summary,key_points"
        fields: List[str] = [f.strip() for f in fields_csv_final.split(",") if f.strip()] if call_ai else None

        def fmt_eta(sec_left: float) -> str:
            sec_left = max(0, sec_left)
            m, s = divmod(int(sec_left), 60)
            return f"{m}m {s}s" if m else f"{s}s"

        def cb(ev: dict):
            stage = ev.get("stage")
            i = ev.get("i")
            n = ev.get("n")
            if n and state["total_pages_est"] is None:
                state["total_pages_est"] = n

            # stage label
            stage_labels = {
                "page_start": "Starting page",
                "text_layer": "Reading embedded text layer",
                "pdftotext": "Fallback: pdftotext",
                "ocrmypdf": "Rebuilding text layer via OCRmyPDF",
                "preprocess": "Preprocessing (denoise/threshold/deskew)",
                "ocr": "OCR (multi-pass)",
                "ocr_try": f"OCR try (langs={ev.get('langs')}, psm={ev.get('psm')})",
                "tables": "Extracting tables",
                "ai": "Extracting AI fields",
                "page_done": "Finishing page",
            }
            if stage in stage_labels:
                stage_line.write(f"**Stage:** {stage_labels[stage]}  {(f'— page {i}/{n}' if i and n else '')}")

            # progress & ETA
            if stage == "page_done":
                state["done_pages"] += 1
                sec = float(ev.get("sec", 0.0))
                if sec > 0:
                    page_times.append(sec)

            if state["total_pages_est"]:
                frac = state["done_pages"] / state["total_pages_est"]
                bar.progress(min(1.0, frac))
                elapsed = time.perf_counter() - start_all
                avg = (sum(page_times) / len(page_times)) if page_times else None
                remain = (state["total_pages_est"] - state["done_pages"]) * avg if (avg and state["total_pages_est"]) else None
                if avg is not None and remain is not None:
                    eta_line.write(
                        f"**Progress:** {state['done_pages']}/{state['total_pages_est']}  •  "
                        f"**Elapsed:** {elapsed:.1f}s  •  **ETA:** {fmt_eta(remain)}"
                    )

            # optional live log (only for heavy stages)
            if stage in ("ocrmypdf","preprocess","ocr","tables","ai","ocr_try"):
                with log:
                    st.write(f"- {stage_labels.get(stage, stage)} (page {i}/{n})")

        # run
        doc = extract_pagewise(
            fb,
            up.name,
            api_key=(api_key if call_ai else ""),
            model=model,
            fields=fields,
            aggressive_ocr=aggressive_ocr,
            try_ocrmypdf=try_ocrmypdf,
            progress_cb=cb,  # <—— hook
        )

        # finalize progress
        bar.progress(1.0)
        total_elapsed = time.perf_counter() - start_all
        eta_line.write(f"**Completed in {total_elapsed:.1f}s**")
        st.toast(f"Finished {up.name}", icon="✅")

        # JSON
        json_bytes = json.dumps(doc.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Download JSON (page-wise)", json_bytes,
                           file_name=up.name.rsplit(".",1)[0] + "_pagewise.json",
                           mime="application/json")

        # Excel
        xbuf = io.BytesIO()
        sheets = build_excel_sheets(doc)
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
            for sname, df in sheets.items():
                df.to_excel(xw, index=False, sheet_name=sname[:31])
        xbuf.seek(0)
        st.download_button("⬇️ Download Excel (page-wise + normalized)", xbuf.getvalue(),
                           file_name=up.name.rsplit(".",1)[0] + "_pagewise.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Preview
        if doc.rules:
            st.info(f"Detected rules: {doc.rules}")
        if doc.normalized:
            fs = len(doc.normalized.get("first_schedule_items", []))
            ss = len(doc.normalized.get("second_schedule_items", []))
            st.success(f"Normalized rows — First: {fs}   Second: {ss}")

        for p in doc.pages:
            with st.expander(f"Page {p.page_index}"):
                st.text_area("Text", p.text, height=220)
                if p.kv_fields:
                    kv_rows = [{"key": k, **fv.model_dump()} for k, fv in p.kv_fields.items()]
                    st.dataframe(pd.DataFrame(kv_rows))
                if p.tables:
                    for t in p.tables:
                        cols = [c.name for c in t.columns]
                        st.caption(f"Table {t.name}")
                        st.dataframe(pd.DataFrame(t.rows, columns=cols))

        st.success("Done.")
