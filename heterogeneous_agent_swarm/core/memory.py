from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import time
import numpy as np
from collections import deque


@dataclass
class EpisodicRecord:
    """
    A single record in episodic memory.
    """
    ts: float
    goal: str
    task: str
    summary: str
    success: bool
    score: float
    artifacts: Dict[str, Any]

class MemoryEntry:
    def __init__(self, vector: np.ndarray, metadata: Dict[str, Any]):
        self.vector = vector
        self.metadata = metadata
        self.similarity = 0.0  # Filled during retrieval

class WorkingMemory:
    """
    Short-term key-value storage for immediate context.
    """
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def set(self, k: str, v: Any) -> None:
        """Set a value in working memory."""
        self.store[k] = v

    def get(self, k: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        return self.store.get(k, default)

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current memory state."""
        return dict(self.store)


class EpisodicMemory:
    """
    Simple episodic memory with cosine similarity retrieval.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add experience to memory."""
        entry = MemoryEntry(vector.copy(), metadata.copy())
        self.buffer.append(entry)

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve top-k similar experiences.
        Uses cosine similarity.
        """
        if not self.buffer:
            return []

        query = query_vector.flatten()
        query_norm = np.linalg.norm(query) + 1e-8

        scored_entries = []
        for entry in self.buffer:
            vec = entry.vector.flatten()
            vec_norm = np.linalg.norm(vec) + 1e-8

            # Cosine similarity
            similarity = np.dot(query, vec) / (query_norm * vec_norm)

            # Add small bonus for recent entries (recency bias)
            recency = 0.0
            if "timestamp" in entry.metadata:
                age = time.time() - entry.metadata["timestamp"]
                recency = 0.1 * np.exp(-age / 100)  # Decay over 100 seconds

            entry.similarity = similarity + recency
            scored_entries.append(entry)

        # Sort by similarity (descending)
        scored_entries.sort(key=lambda x: x.similarity, reverse=True)

        return scored_entries[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.buffer:
            return {"count": 0}

        rewards = [e.metadata.get("reward", 0) for e in self.buffer]
        return {
            "count": len(self.buffer),
            "avg_reward": np.mean(rewards),
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0
        }


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
        """Add a vector and associated payload."""
        self.items.append((vec, payload))

    def topk(self, vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar items to the query vector."""
        scored = [(self._cos(vec, v), p) for v, p in self.items if len(v) == len(vec)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p | {"similarity": s} for s, p in scored[:k]]
