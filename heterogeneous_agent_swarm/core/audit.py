from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

@dataclass
class AuditRecord:
    ts: float
    event: str
    payload: Dict[str, Any]

class AuditLog:
    def __init__(self):
        self.records: List[AuditRecord] = []

    def emit(self, event: str, payload: Dict[str, Any]) -> None:
        rec = AuditRecord(ts=time.time(), event=event, payload=payload)
        self.records.append(rec)

    def tail(self, n: int = 5) -> List[AuditRecord]:
        return self.records[-n:]
