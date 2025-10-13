import io, os, json, hashlib, time
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




# --------------
# Public API
# --------------


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


kv_fields = {}
if fields and api_key:
kv_raw = _llm_extract_kv(txt, fields, api_key, model=model)
# Wrap into FieldValue entries
kv_fields = {
k: FieldValue(value=v, type=("number" if isinstance(v, (int, float)) else "string"), page=idx1)
for k, v in kv_raw.items()
}


pages.append(PageModel(page_index=idx1, text=txt, tables=tables, kv_fields=kv_fields))


source = DocSource(filename=filename, pages=len(pages), sha256=sha, lang=lang)
doc = CanonicalDoc(doc_id=doc_id, source=source, pages=pages)
return doc


def build_excel_sheets(doc: CanonicalDoc) -> Dict[str, "pandas.DataFrame"]:
import pandas as pd
sheets: Dict[str, pd.DataFrame] = {}


# Text sheet per page
for p in doc.pages:
sname = f"p{p.page_index:02d}_text"[:31]
sheets[sname] = pd.DataFrame({"text": [p.text]})


# KV (if any)
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


# Tables
for t in p.tables:
cols = [c.name for c in t.columns]
df = pd.DataFrame(t.rows, columns=cols)
df.insert(0, "page", p.page_index)
sheets[t.name[:31]] = df


return sheets