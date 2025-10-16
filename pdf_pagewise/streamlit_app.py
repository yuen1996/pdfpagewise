import io
import os
import json
import time
from typing import List
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from page_extractor import extract_pagewise, extract_image, build_excel_sheets, DEFAULT_MODEL

# Load .env twice (repo root + package)
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

st.set_page_config(page_title="PDF/Image → AI-Readable", layout="wide")
st.title("📄→🧠 PDF/Image to AI-Readable JSON/Excel")

with st.sidebar:
    st.header("Settings")
    ocr_langs = st.text_input("Tesseract OCR languages", os.getenv("OCR_LANGS", "chi_sim+chi_tra+eng+msa"))
    dpi = st.slider("Render DPI for OCR pages", 200, 500, int(os.getenv("OCR_DPI", "300")))
    aggressive_ocr = st.checkbox("Aggressive OCR (OpenCV + multi-pass)", value=True)
    try_ocrmypdf = st.checkbox("Try OCRmyPDF first (if installed)", value=True)
    speed_mode = st.checkbox(
        "Speed mode (fast)", value=True,
        help="OCRmyPDF once per file, then minimal OCR passes per page."
    )

    # AI extraction
    raw_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    api_key = (raw_key or "").strip()
    model = st.selectbox("Model", [DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini"], index=0)
    use_ai = st.checkbox("Use AI extraction (per page)", value=bool(api_key))
    fields_default = os.getenv("DEFAULT_AI_FIELDS", "page_summary,key_points")
    fields_csv = st.text_input("Fields to extract (comma-separated)", value=fields_default)

    # quick key sanity hint
    if use_ai:
        if not api_key.startswith("sk-"):
            st.warning("API key format looks unusual. Keys usually start with “sk-”. Check your .env or sidebar input.")
        elif len(api_key) < 30:
            st.warning("API key seems too short. Re-copy it from https://platform.openai.com/account/api-keys")

    st.caption(f"Langs={ocr_langs}  DPI={dpi}  OEM={os.getenv('OCR_OEM','1')}")

files = st.file_uploader("Upload PDFs or Images", type=["pdf","jpg","jpeg","png"], accept_multiple_files=True)
if not files:
    st.info("Upload PDFs or Images to begin.")
    st.stop()

if st.button("Process (Page-by-Page)", type="primary"):
    os.environ["OCR_LANGS"] = ocr_langs
    os.environ["OCR_DPI"] = str(dpi)

    for up in files:
        st.subheader(f"📄 {up.name}")
        fb = up.read()

        bar = st.progress(0)
        stage_line = st.empty()
        eta_line = st.empty()
        log = st.expander("Details (log)")
        page_times: List[float] = []
        start_all = time.perf_counter()
        state = {"done_pages": 0, "total_pages_est": None}

        call_ai = use_ai and bool(api_key)
        fields_csv_final = (fields_csv or "").strip() or ("page_summary,key_points" if call_ai else "")
        fields: List[str] = [f.strip() for f in fields_csv_final.split(",") if f.strip()] if call_ai else None

        def fmt_eta(sec_left: float) -> str:
            sec_left = max(0, sec_left)
            m, s = divmod(int(sec_left), 60)
            return f"{m}m {s}s" if m else f"{s}s"

        def cb(ev: dict):
            stage = ev.get("stage")
            i, n = ev.get("i"), ev.get("n")
            if n and state["total_pages_est"] is None:
                state["total_pages_est"] = n

            labels = {
                "page_start": "Starting page",
                "pdftotext": "Fallback: pdftotext",
                "ocrmypdf": "Rebuilding text layer via OCRmyPDF (once)",
                "preprocess": "Preprocessing (denoise/threshold/deskew)",
                "ocr": "OCR (multi-pass, pruned)",
                "ocr_try": f"OCR try (langs={ev.get('langs')}, psm={ev.get('psm')})",
                "tables": "Extracting tables",
                "ai": "Extracting AI fields",
                "page_done": "Finishing page",
            }
            if stage in labels:
                stage_line.write(f"**Stage:** {labels[stage]}  {(f'— page {i}/{n}' if i and n else '')}")

            if stage == "page_done":
                state["done_pages"] += 1
                sec = float(ev.get("sec", 0.0))
                if sec > 0: page_times.append(sec)

            if state["total_pages_est"]:
                frac = state["done_pages"] / state["total_pages_est"]
                bar.progress(min(1.0, frac))
                elapsed = time.perf_counter() - start_all
                avg = (sum(page_times)/len(page_times)) if page_times else None
                remain = (state["total_pages_est"] - state["done_pages"]) * avg if (avg and state["total_pages_est"]) else None
                if avg is not None and remain is not None:
                    eta_line.write(f"**Progress:** {state['done_pages']}/{state['total_pages_est']}  •  "
                                   f"**Elapsed:** {elapsed:.1f}s  •  **ETA:** {fmt_eta(remain)}")

            if stage in ("ocrmypdf","preprocess","ocr","tables","ai","ocr_try"):
                with log:
                    st.write(f"- {labels.get(stage, stage)} (page {i}/{n})")

        ext = Path(up.name).suffix.lower()
        try:
            if ext == ".pdf":
                doc = extract_pagewise(
                    fb, up.name,
                    api_key=(api_key if call_ai else ""),
                    model=model,
                    fields=fields,
                    aggressive_ocr=aggressive_ocr,
                    try_ocrmypdf=try_ocrmypdf,
                    speed_mode=speed_mode,
                    progress_cb=cb,
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
                )
        except Exception as e:
            # Last-resort guard: show a nice message and continue with next file
            st.error(dedent(f"""\
                🛑 Processing failed.
                {e}

                Tips:
                • Recheck your OpenAI API key (sidebar or .env).
                • Toggle off “Use AI extraction” to run without LLM.
                • See logs: journalctl -u pdf-pagewise -n 100 --no-pager
            """))
            continue

        # If extractor flagged an AI error/warning, show it nicely
        if getattr(doc, "meta", None):
            if doc.meta.get("ai_error"):
                st.error("OpenAI: " + str(doc.meta["ai_error"]))
                st.info("Continuing without AI. OCR/text/tables/Excel still generated.")
            elif doc.meta.get("ai_warning"):
                st.warning(str(doc.meta["ai_warning"]))

        # Downloads
        json_bytes = json.dumps(doc.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Download JSON (page-wise)", json_bytes,
                           file_name=up.name.rsplit(".",1)[0] + "_pagewise.json",
                           mime="application/json")

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

        if getattr(doc, "meta", None) and (doc.meta.get("doc_summary") or doc.meta.get("doc_key_points")):
            st.subheader("🧠 Document Summary")
            if doc.meta.get("doc_summary"):
                st.write(doc.meta["doc_summary"])
            if doc.meta.get("doc_key_points"):
                st.write("**Key points:**")
                for kp in doc.meta["doc_key_points"]:
                    st.write(f"- {kp}")

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
