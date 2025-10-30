from __future__ import annotations

from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


class FieldValue(BaseModel):
    """
    Generic key/value extracted by an LLM or rules engine.
    """
    value: Any
    type: Literal["string", "number", "date", "enum", "boolean", "object", "array"] = "string"
    unit: Optional[str] = None
    confidence: Optional[float] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    enum: Optional[List[str]] = None


class TableColumn(BaseModel):
    name: str
    type: Literal["string", "number", "date", "enum", "boolean"] = "string"
    unit: Optional[str] = None


class TableModel(BaseModel):
    """
    A simple table model: a list of columns and 2D rows.
    """
    name: str = "table"
    columns: List[TableColumn] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)


class OCRLine(BaseModel):
    """
    A single OCR line with confidence and polygon (clockwise points).
    """
    text: str
    conf: Optional[float] = None
    poly: List[List[float]] = Field(default_factory=list)


class PageModel(BaseModel):
    """
    One page of text/tables plus optional AI-extracted fields.
    """
    page_index: int
    text: str = ""
    tables: List[TableModel] = Field(default_factory=list)
    kv_fields: Dict[str, FieldValue] = Field(default_factory=dict)
    lang: Optional[str] = None
    ocr: bool = False
    lines: List[OCRLine] = Field(default_factory=list)


class DocSource(BaseModel):
    """
    Information about the original source document.
    """
    filename: Optional[str] = None
    pages: Optional[int] = None
    sha256: Optional[str] = None
    lang: Optional[str] = None
    ingested_at: Optional[str] = None


class CanonicalDoc(BaseModel):
    """
    The full, canonical representation produced by this app.
    """
    doc_id: str
    doc_type: str = "unknown"
    schema_version: str = "1.1.0"
    source: DocSource
    pages: List[PageModel]
    # Optional metadata for downstream systems
    rules: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    normalized: Optional[Dict[str, Any]] = None
