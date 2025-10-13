import io, os, json
from typing import List


import pandas as pd
import streamlit as st
from dotenv import load_dotenv


from schema import CanonicalDoc
from page_extractor import extract_pagewise, build_excel_sheets, DEFAULT_MODEL


load_dotenv()
st.set_page_config(page_title="Page-by-Page PDF OCR → JSON → Excel", layout="wide")
st.title("Page-by-Page PDF OCR → JSON → Excel")


with st.sidebar:
st.header("Settings")
api_key = st.text_input(
"OpenAI API Key (optional)",
type="password",
value=os.getenv("OPENAI_API_KEY", ""),
)
model = st.selectbox("Model", [DEFAULT_MODEL, "gpt-4o", "gpt-4o-mini"], index=0)
fields_csv = st.text_input(
"Fields to extract per page (comma-separated)",
value="", # leave empty to skip LLM extraction
help="If provided and API key set, the app will extract these fields per page via AI.",
)


files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if not files:
st.info("Upload PDFs to begin.")
st.stop()


if st.button("Process (Page-by-Page)", type="primary"):
for up in files:
st.subheader(f"📄 {up.name}")
fb = up.read()


fields: List[str] = [f.strip() for f in fields_csv.split(",") if f.strip()]
doc = extract_pagewise(
fb,
up.name,
api_key=api_key if (api_key and fields) else "",
model=model,
fields=fields if fields else None,
)


# JSON output
json_bytes = json.dumps(doc.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")
st.download_button(
"⬇️ Download JSON (page-wise)",
json_bytes,
file_name=up.name.rsplit(".", 1)[0] + "_pagewise.json",
mime="application/json",
)


# Excel output (one sheet per page: text, kv, and each table)
xbuf = io.BytesIO()
sheets = build_excel_sheets(doc)
with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
for sname, df in sheets.items():
df.to_excel(xw, index=False, sheet_name=sname)
xbuf.seek(0)
st.download_button(
"⬇️ Download Excel (page-wise)",
xbuf.getvalue(),
file_name=up.name.rsplit(".", 1)[0] + "_pagewise.xlsx",
mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


# Visual review: page expanders
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