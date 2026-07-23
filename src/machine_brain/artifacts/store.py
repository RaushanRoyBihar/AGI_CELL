"""Layer 10 — long-term artifact memory. Local files in the prototype;
MinIO/S3 at fleet scale, behind the same ArtifactStore interface.
PostgreSQL (here: SQLite, same schema) stores hash/ownership/timestamp/
retention rows — the artifact content itself is addressed by hash, never
duplicated into canonical storage.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    content_hash TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    owner_robot_id TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at REAL NOT NULL,
    retention_policy TEXT NOT NULL DEFAULT 'default'
);
"""


@dataclass(frozen=True)
class ArtifactRef:
    content_hash: str
    kind: str
    size_bytes: int


class ArtifactStore(ABC):
    @abstractmethod
    def put(self, kind: str, owner_robot_id: str, data: bytes, retention_policy: str = "default") -> ArtifactRef: ...

    @abstractmethod
    def get(self, content_hash: str) -> bytes | None: ...


class LocalFileArtifactStore(ArtifactStore):
    def __init__(self, base_dir: str, index_db_path: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.conn = sqlite3.connect(index_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def put(self, kind: str, owner_robot_id: str, data: bytes, retention_policy: str = "default") -> ArtifactRef:
        content_hash = hashlib.sha256(data).hexdigest()
        path = os.path.join(self.base_dir, content_hash)
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(data)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO artifacts(content_hash, kind, owner_robot_id, size_bytes, created_at, retention_policy) "
                "VALUES (?,?,?,?,?,?)",
                (content_hash, kind, owner_robot_id, len(data), time.time(), retention_policy),
            )
        return ArtifactRef(content_hash=content_hash, kind=kind, size_bytes=len(data))

    def get(self, content_hash: str) -> bytes | None:
        path = os.path.join(self.base_dir, content_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()


class MinioArtifactStore(ArtifactStore):
    """Optional adapter — activated via config/adapters.yaml `artifacts:
    minio`. Requires network access to a MinIO/S3-compatible endpoint; not
    used by default."""

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str) -> None:
        try:
            from minio import Minio  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "MinioArtifactStore requires the 'minio' package. This adapter is optional — "
                "the system runs on LocalFileArtifactStore without it."
            ) from e
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key)
        self._bucket = bucket

    def put(self, kind, owner_robot_id, data, retention_policy="default"):
        raise NotImplementedError("wire real MinIO put_object when deployed at fleet scale")

    def get(self, content_hash):
        raise NotImplementedError
