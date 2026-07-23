"""Layer 5 — associative memory. Qdrant Edge locally / Qdrant server at
fleet scale, per spec — but Qdrant is an optional dependency, not
installed by default. `LocalVectorIndex` implements the same
`AssociativeIndex` interface with plain NumPy cosine similarity, so the
system's recall path works with zero external services. `QdrantIndex` is
the same interface backed by the real qdrant-client, activated only by
config.

Hard rule enforced here, not just documented: every `search()` result
carries a `canonical_id` and nothing else authoritative — callers MUST
resolve that id against episodic/working-memory storage before acting on
it. This module never returns "truth", only candidates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    canonical_id: str
    score: float


class AssociativeIndex(ABC):
    @abstractmethod
    def upsert(self, canonical_id: str, vector: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[Candidate]: ...

    @abstractmethod
    def remove(self, canonical_id: str) -> None: ...

    @abstractmethod
    def rebuild(self, items: list[tuple[str, np.ndarray]]) -> None:
        """Wipe and reindex from canonical data — the standard recovery
        path for a corrupted/diverged derived index."""


class LocalVectorIndex(AssociativeIndex):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=float)
        self._id_to_row: dict[str, int] = {}

    def upsert(self, canonical_id: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=float).reshape(1, -1)
        if canonical_id in self._id_to_row:
            row = self._id_to_row[canonical_id]
            self._vectors[row] = vector
            return
        self._id_to_row[canonical_id] = len(self._ids)
        self._ids.append(canonical_id)
        self._vectors = np.vstack([self._vectors, vector])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[Candidate]:
        if len(self._ids) == 0:
            return []
        q = np.asarray(query_vector, dtype=float).reshape(1, -1)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        v_norm = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9)
        scores = (v_norm @ q_norm.T).ravel()
        order = np.argsort(-scores)[:top_k]
        return [Candidate(canonical_id=self._ids[i], score=float(scores[i])) for i in order]

    def remove(self, canonical_id: str) -> None:
        if canonical_id not in self._id_to_row:
            return
        self.rebuild([(cid, self._vectors[row]) for cid, row in self._id_to_row.items() if cid != canonical_id])

    def rebuild(self, items: list[tuple[str, np.ndarray]]) -> None:
        self._ids = []
        self._id_to_row = {}
        self._vectors = np.zeros((0, self.dim), dtype=float)
        for cid, vec in items:
            self.upsert(cid, vec)


class QdrantIndex(AssociativeIndex):
    """Optional adapter — same AssociativeIndex interface, activated via
    config/adapters.yaml `associative: qdrant`. Requires the `qdrant`
    extra and a running Qdrant instance; not used by default."""

    def __init__(self, url: str, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "QdrantIndex requires the 'qdrant-client' package (pip install machine_brain[qdrant]). "
                "This adapter is optional — the system runs on LocalVectorIndex without it."
            ) from e
        self._client = QdrantClient(url=url)
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, canonical_id, vector):
        from qdrant_client.models import PointStruct  # type: ignore
        self._client.upsert(self._collection, points=[PointStruct(id=canonical_id, vector=vector.tolist())])

    def search(self, query_vector, top_k=5):
        hits = self._client.search(self._collection, query_vector=query_vector.tolist(), limit=top_k)
        return [Candidate(canonical_id=str(h.id), score=h.score) for h in hits]

    def remove(self, canonical_id):
        self._client.delete(self._collection, points_selector=[canonical_id])

    def rebuild(self, items):
        for cid, vec in items:
            self.upsert(cid, vec)
