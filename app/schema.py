# app/schema.py
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel


class Document(BaseModel):
    content: str
    source: str = "unknown"
    origin: Literal["vector", "web"] = "vector"
    page: Optional[int] = None

    # backward-compat shim — agent/nodes.py uses .page_content
    @property
    def page_content(self) -> str:
        return self.content