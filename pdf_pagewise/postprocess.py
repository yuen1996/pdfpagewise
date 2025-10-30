# postprocess.py
import re
from typing import Dict, Any, List, Optional

# ---------- text cleanup ----------

_SPACES_MULTI = re.compile(r"[ \t]{2,}")
_SOFT_HYPHEN = "\u00ad"
CJK = r"[\u4E00-\u9FFF]"

def clean_description(s: str) -> str:
    """Light cleanup for description cells."""
    if s is None:
        return s
    s = s.replace(_SOFT_HYPHEN, "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(_SPACES_MULTI, " ", s)
    # join broken CJK lines
    s = re.sub(rf"({CJK})\s+\n\s*({CJK})", r"\1\2", s)
    s = s.strip()
    return s

# ---------- normalization helpers ----------

def _first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        vs = str(v).strip()
        if vs:
            return vs
    return None

def _maybe_rate(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if re.search(r"\b\d+(\.\d+)?\s*%\b", s):
        return re.search(r"\d+(\.\d+)?\s*%", s).group(0)
    if s.isdigit():
        return f"{s}%"
    return None

# ---------- normalization ----------

def normalize_first_schedule_items(pages: List[Any], rate_percent_text: Optional[str], doc_id: str) -> List[Dict[str, Any]]:
    """
    Heuristic normalizer for Malaysia Sales Tax First Schedule-like tables.
    Produces rows with keys: item_no, hs_code, description, rate, appendix, page.
    """
    out: List[Dict[str, Any]] = []
    default_rate = rate_percent_text

    for p in pages:
        if not getattr(p, "tables", None):
            continue
        for t in p.tables:
            cols = [c.name.lower() for c in t.columns]
            # Try to find candidate columns
            try:
                name_to_idx = {c.lower(): i for i, c in enumerate([c.name for c in t.columns])}
            except Exception:
                name_to_idx = {f"col_{i+1}": i for i in range(len(t.columns))}

            # very loose heuristics
            idx_code = next((i for i, c in enumerate(cols) if "hs" in c or "tariff" in c or "code" in c), None)
            idx_desc = next((i for i, c in enumerate(cols) if "desc" in c), None)
            idx_item = next((i for i, c in enumerate(cols) if "item" in c), None)
            idx_rate = next((i for i, c in enumerate(cols) if "rate" in c or "%" in c), None)
            idx_appx = next((i for i, c in enumerate(cols) if "appendix" in c), None)

            for r in t.rows:
                def col(i): 
                    try:
                        return r[i]
                    except Exception:
                        return None

                row = {
                    "doc_id": doc_id,
                    "page": getattr(p, "page_index", None),
                    "item_no": _first_nonempty(col(idx_item)),
                    "hs_code": _first_nonempty(col(idx_code)),
                    "description": clean_description(_first_nonempty(col(idx_desc)) or ""),
                    "rate": _maybe_rate(col(idx_rate)) or default_rate,
                    "appendix": _first_nonempty(col(idx_appx)),
                }

                # minimal filter: must have either hs_code or description
                if row["hs_code"] or row["description"]:
                    out.append(row)

    return out

def normalize_second_schedule_items(pages: List[Any], doc_id: str) -> List[Dict[str, Any]]:
    """
    Placeholder normalizer for 'Second Schedule'-style lists.
    Extracts simpler 'exempted items' if there are description-only tables.
    """
    out: List[Dict[str, Any]] = []
    for p in pages:
        if not getattr(p, "tables", None):
            continue
        for t in p.tables:
            cols = [c.name.lower() for c in t.columns]
            # If only one or two columns and one looks like description
            if len(cols) <= 2 and any("desc" in c or "item" in c or "goods" in c for c in cols):
                for r in t.rows:
                    desc = clean_description(_first_nonempty(*r) or "")
                    if desc:
                        out.append({
                            "doc_id": doc_id,
                            "page": getattr(p, "page_index", None),
                            "description": desc
                        })
    return out
