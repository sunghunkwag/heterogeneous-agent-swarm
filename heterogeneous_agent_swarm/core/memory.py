from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math
import time


@dataclass
class EpisodicRecord:
    ts: float
    goal: str
    task: str
    summary: str
    success: bool
    score: float
    artifacts: Dict[str, Any]


class WorkingMemory:
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def set(self, k: str, v: Any) -> None:
        self.store[k] = v

    def get(self, k: str, default=None):
        return self.store.get(k, default)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.store)


class EpisodicMemory:
    def __init__(self, max_records: int = 2000):
        self.max_records = max_records
        self.records: List[EpisodicRecord] = []

    def add(self, rec: EpisodicRecord) -> None:
        self.records.append(rec)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]


class SemanticMemory:
    """
    Minimal vector store without external deps.
    You can later replace with FAISS or a real embedding model.
    """
    def __init__(self):
        self.items: List[Tuple[List[float], Dict[str, Any]]] = []

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        na = math.sqrt(sum(x*x for x in a)) + 1e-9
        nb = math.sqrt(sum(x*x for x in b)) + 1e-9
        return sum(x*y for x, y in zip(a, b)) / (na * nb)

    def add(self, vec: List[float], payload: Dict[str, Any]) -> None:
        self.items.append((vec, payload))

    def topk(self, vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        scored = [(self._cos(vec, v), p) for v, p in self.items if len(v) == len(vec)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p | {"similarity": s} for s, p in scored[:k]]
