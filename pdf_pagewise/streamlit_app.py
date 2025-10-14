import io
import os
import json
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

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")

    # API key: prefer what user types; fall back to .env
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Needed only for AI field extraction (summaries, custom fields).",
    )

    model = st.selectbox("Model", [DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini"], index=0)

    use_ai = st.checkbox(
        "Use AI extraction (per page)",
        value=bool(api_key or os.getenv("OPENAI_API_KEY")),
        help="If on, the app will call OpenAI per page to extract fields (below).",
    )

    fields_default = os.getenv("DEFAULT_AI_FIELDS", "page_summary,key_points")
    fields_csv = st.text_input(
        "Fields to extract (comma-separated)",
        value=fields_default,
        help="Examples: page_summary,key_points,rate_clause,document_title . "
             "If left blank while 'Use AI extraction' is ON, defaults to page_summary,key_points.",
    )

    # Debug view of OCR env (safe to remove later)
    st.caption(
        f"OCR_LANGS={os.getenv('OCR_LANGS')}  DPI={os.getenv('OCR_DPI')}  "
        f"PSM={os.getenv('OCR_PSM')}  OEM={os.getenv('OCR_OEM')}"
    )

# ---- File upload ----
files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if not files:
    st.info("Upload PDFs to begin.")
    st.stop()

# ---- Process button ----
if st.button("Process (Page-by-Page)", type="primary"):
    for up in files:
        st.subheader(f"📄 {up.name}")
        fb = up.read()

        # Decide whether to call OpenAI
        call_ai = use_ai and bool(api_key)
        fields_csv_final = (fields_csv or "").strip()
        if call_ai and not fields_csv_final:
            fields_csv_final = "page_summary,key_points"

        fields: List[str] = [f.strip() for f in fields_csv_final.split(",") if f.strip()] if call_ai else None

        doc = extract_pagewise(
            fb,
            up.name,
            api_key=(api_key if call_ai else ""),
            model=model,
            fields=fields,
        )

        # JSON download
        json_bytes = json.dumps(doc.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON (page-wise)",
            json_bytes,
            file_name=up.name.rsplit(".", 1)[0] + "_pagewise.json",
            mime="application/json",
        )

        # Excel download (page-wise + normalized + rules)
        xbuf = io.BytesIO()
        sheets = build_excel_sheets(doc)
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
            for sname, df in sheets.items():
                df.to_excel(xw, index=False, sheet_name=sname[:31])  # Excel max 31 chars
        xbuf.seek(0)
        st.download_button(
            "⬇️ Download Excel (page-wise + normalized)",
            xbuf.getvalue(),
            file_name=up.name.rsplit(".", 1)[0] + "_pagewise.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Quick preview: rules + normalized counts
        if doc.rules:
            st.info(f"Detected rules: {doc.rules}")
        if doc.normalized:
            fs = len(doc.normalized.get("first_schedule_items", []))
            ss = len(doc.normalized.get("second_schedule_items", []))
            st.success(f"Normalized rows — First: {fs}   Second: {ss}")

        # Visual review per page (includes KV if AI ran)
        for p in doc.pages:
            with st.expander(f"Page {p.page_index}"):
                st.text_area("Text", p.text, height=200)
                if p.kv_fields:
                    kv_rows = [{"key": k, **fv.model_dump()} for k, fv in p.kv_fields.items()]
                    st.dataframe(pd.DataFrame(kv_rows))
                if p.tables:
                    for t in p.tables:
                        cols = [c.name for c in t.columns]
                        st.caption(f"Table {t.name}")
                        st.dataframe(pd.DataFrame(t.rows, columns=cols))

        st.success("Done.")
