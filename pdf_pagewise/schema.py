from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field

class FieldValue(BaseModel):
    value: Any
    type: Literal["string", "number", "date", "enum", "boolean", "object", "array"]
    unit: Optional[str] = None
    confidence: Optional[float] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    enum: Optional[List[str]] = None

class TableColumn(BaseModel):
    name: str
    type: Literal["string", "number", "date", "enum", "boolean"]
    unit: Optional[str] = None

class TableModel(BaseModel):
    name: str
    page: Optional[int] = None
    columns: List[TableColumn]
    rows: List[List[Any]]

class PageModel(BaseModel):
    page_index: int  # 1-based
    text: str
    tables: List[TableModel] = Field(default_factory=list)
    kv_fields: Dict[str, FieldValue] = Field(default_factory=dict)
    # Helpful metadata:
    lang: Optional[str] = None         # "ms" | "en" | "zh" | None
    ocr: Optional[bool] = None         # True if OCR used for this page

class DocSource(BaseModel):
    filename: str
    pages: int
    sha256: Optional[str] = None
    lang: Optional[str] = None
    ingested_at: Optional[str] = None

class CanonicalDoc(BaseModel):
    doc_id: str
    doc_type: str = "unknown"
    schema_version: str = "1.0.0"
    source: DocSource
    pages: List[PageModel]
    # For AI/DB use:
    rules: Optional[Dict[str, Any]] = None       # e.g., {"first_schedule_rate": "5%"}
    meta: Optional[Dict[str, Any]] = None        # e.g., {"cover_text": "..."}
    normalized: Optional[Dict[str, Any]] = None  # e.g., {"first_schedule_items": [...]}
