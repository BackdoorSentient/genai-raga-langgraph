# app/agent/document.py
from typing import Dict, Optional

class AgentDocument:
    def __init__(self, page_content: str, metadata: Dict = None,source: Optional[str] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.source = source
