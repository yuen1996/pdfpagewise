# postprocess.py
import re
from typing import Dict, Any, List, Optional

# ---------- text cleanup ----------

_SPACES_MULTI = re.compile(r"[ \t]{2,}")
_SOFT_HYPHEN = "\u00ad"  # sometimes appears in PDFs

def _join_small_chunks(token: str) -> str:
    """
    Join weird inner spaces like 'Die sel fu el' -> 'Dieselfuel' (then fix common words).
    Heuristic: if token contains inner spaces and most chunks are <=3, join.
    """
    parts = token.split(" ")
    if len(parts) <= 1:
        return token
    short = sum(1 for p in parts if 1 <= len(p) <= 3)
    if short >= len(parts) - 1 and len("".join(parts)) >= 5:
        return "".join(parts)
    return token

COMMON_FIXES = {
    "Dieselfuel": "Diesel fuel",
    "dieselfuel": "diesel fuel",
    "petro leum": "petroleum",
    "li quid": "liquid",
    "na tu ral": "natural",
}

def clean_description(s: str) -> str:
    if not s:
        return s
    t = s.replace(_SOFT_HYPHEN, "")
    # remove hyphen breaks like "some-\nthing" or "some- thing"
    t = re.sub(r"-\s*\n\s*", "", t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    # collapse linebreaks to spaces
    t = re.sub(r"\s*\n\s*", " ", t)
    # collapse multiple spaces
    t = _SPACES_MULTI.sub(" ", t).strip()

    # fix inner-space chopped words per token
    tokens = [_join_small_chunks(tok) for tok in t.split(" ")]
    t = " ".join(tokens)

    # targeted fixes
    for bad, good in COMMON_FIXES.items():
        t = t.replace(bad, good)
    return t.strip()

# ---------- optional bullets from long descriptions ----------

def description_to_bullets(raw: str) -> List[str]:
    """
    Heuristics to split long clauses into bullets.
    We prefer explicit line-bullets ('\n- ') but fall back to ' - ' separators
    when there are 2+ occurrences.
    """
    if not raw:
        return []
    r = raw.replace("\r", "")
    if "\n- " in r:
        items = [x.strip(" -•\t") for x in r.split("\n- ")[1:]]
        return [clean_description(x) for x in items if len(x) >= 3]
    # fallback: many " - " separators in a single line
    if r.count(" - ") >= 2:
        parts = [p.strip(" -•\t") for p in r.split(" - ")]
        return [clean_description(p) for p in parts if len(p) >= 3]
    return []

# ---------- HS helpers ----------

HS_FULL = re.compile(r"^\d{4}\.\d{2}\.\d{2}\s+\d{2}$")  # e.g., 0208.10.00 00

def clean_hs_subheading(raw: str) -> str:
    """Remove inner spaces but keep dots: '0208.10.00 00' -> '0208.10.0000'."""
    return re.sub(r"\s+", "", raw or "").strip()

def hs_heading(raw: str) -> Optional[str]:
    r = (raw or "").strip()
    return r or None

# ---------- rate parsing (Second Schedule) ----------

RM = re.compile(r"RM\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PER_UNIT = re.compile(r"\bper\s+(kg|kilogram|g|gram|litre|liter|l|m3|tonne|t)\b", re.IGNORECASE)

UNIT_NORMALIZE = {
    "kilogram": "kg",
    "g": "g",
    "gram": "g",
    "litre": "litre",
    "liter": "litre",
    "l": "litre",
    "m3": "m3",
    "tonne": "t",
    "t": "t",
    "kg": "kg",
}

def parse_rate_text(s: str) -> Dict[str, Any]:
    """
    'RM0.40 per litre' -> {'currency':'MYR','specific_rate_value':0.40,'specific_rate_unit':'litre'}
    returns {} if not parsable.
    """
    if not s:
        return {}
    m = RM.search(s)
    if not m:
        return {}
    value = float(m.group(1))
    unit = None
    u = PER_UNIT.search(s)
    if u:
        unit = UNIT_NORMALIZE.get(u.group(1).lower(), u.group(1).lower())
    return {"currency": "MYR", "specific_rate_value": value, "specific_rate_unit": unit}

# ---------- normalization ----------

def normalize_first_schedule_items(pages: List[Any], rate_percent_text: Optional[str], doc_id: str) -> List[Dict[str, Any]]:
    """
    From First Schedule tables (Heading/Subheading/Description), produce clean rows:
    - fill-forward heading
    - split multi-line subheadings
    - attach rate_percent (numeric) from top-level rule '5%'
    - add cleaned HS fields + optional bullets
    """
    rows: List[Dict[str, Any]] = []
    current_heading: Optional[str] = None
    rate_num = None
    if rate_percent_text and rate_percent_text.endswith("%"):
        try:
            rate_num = float(rate_percent_text[:-1])
        except Exception:
            rate_num = None

    for p in pages:
        for t in getattr(p, "tables", []):
            if not t.columns or len(t.columns) < 3:
                continue
            if not t.columns[0].name.lower().startswith("heading"):
                continue

            for raw in t.rows:
                heading, subcell, desc = (raw + ["", "", ""])[:3]
                if heading and str(heading).strip():
                    current_heading = str(heading).strip()
                if not subcell:
                    continue

                subs = [s.strip() for s in str(subcell).splitlines() if s.strip()]
                subs = [s for s in subs if HS_FULL.match(s)]
                if not subs:
                    continue

                bullets = description_to_bullets(str(desc or ""))
                for code in subs:
                    hs_clean = clean_hs_subheading(code)
                    rows.append({
                        "doc_id": doc_id,
                        "page_index": p.page_index,
                        "schedule": "First",
                        "rate_percent": rate_num,              # e.g., 5.0
                        "raw_heading": current_heading,
                        "raw_subheading": code,
                        "hs_chapter_heading": hs_heading(current_heading),
                        "hs_subheading": hs_clean,
                        "description": clean_description(str(desc or "")),
                        **({"bullets": bullets} if bullets else {}),
                        "split_from_multiline": len(subs) > 1,
                    })
    return rows

def _is_second_schedule_header(names: List[str]) -> bool:
    if len(names) < 4:
        return False
    h = [n.lower() for n in names[:4]]
    return ("heading" in h[0]) and ("subheading" in h[1]) and ("description" in h[2]) and ("rate" in h[3])

def normalize_second_schedule_items(pages: List[Any], doc_id: str) -> List[Dict[str, Any]]:
    """
    From tables with Rate column ('Rate of tax (4)'), produce rows with parsed rate.
    """
    rows: List[Dict[str, Any]] = []
    current_heading: Optional[str] = None

    for p in pages:
        for t in getattr(p, "tables", []):
            cols = t.columns or []
            names = [c.name for c in cols]
            if not _is_second_schedule_header(names):
                continue

            for raw in t.rows:
                heading, subcell, desc, rate = (raw + ["", "", "", ""])[:4]
                if heading and str(heading).strip():
                    current_heading = str(heading).strip()
                if not subcell:
                    continue

                subs = [s.strip() for s in str(subcell).splitlines() if s.strip()]
                subs = [s for s in subs if HS_FULL.match(s)]
                if not subs:
                    continue

                rate_info = parse_rate_text(str(rate or ""))
                bullets = description_to_bullets(str(desc or ""))

                for code in subs:
                    hs_clean = clean_hs_subheading(code)
                    rows.append({
                        "doc_id": doc_id,
                        "page_index": p.page_index,
                        "schedule": "Second",
                        "raw_heading": current_heading,
                        "raw_subheading": code,
                        "hs_chapter_heading": hs_heading(current_heading),
                        "hs_subheading": hs_clean,
                        "description": clean_description(str(desc or "")),
                        **({"bullets": bullets} if bullets else {}),
                        "raw_rate_text": str(rate or "").strip(),
                        **rate_info,  # currency, specific_rate_value, specific_rate_unit (if present)
                        "split_from_multiline": len(subs) > 1,
                    })
    return rows
