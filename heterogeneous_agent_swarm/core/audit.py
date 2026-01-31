from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import time


@dataclass
class AuditRecord:
    """
    Single entry in the audit log.

    Args:
        ts: Timestamp of the event.
        event: Event name/type.
        payload: Detailed data associated with the event.
    """
    ts: float
    event: str
    payload: Dict[str, Any]


class AuditLog:
    """
    Immutable ledger of system events for debugging and analysis.
    """
    def __init__(self):
        """Initialize the audit log."""
        self.records: List[AuditRecord] = []

    def emit(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Record a new event.

        Args:
            event: Name of the event.
            payload: Data dictionary for the event.
        """
        rec = AuditRecord(ts=time.time(), event=event, payload=payload)
        self.records.append(rec)

    def tail(self, n: int = 5) -> List[AuditRecord]:
        """
        Get the most recent records.

        Args:
            n: Number of records to retrieve.

        Returns:
            List of the last n AuditRecords.
        """
        return self.records[-n:]
