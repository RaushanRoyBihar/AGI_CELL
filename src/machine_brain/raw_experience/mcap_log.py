"""Layer 2 — raw experience. Real deployments use ROS 2 rosbag2 + MCAP; the
`mcap` Python package is not installed in this environment (no network
access assumed here), so this module implements a self-contained
"mcap_lite" format with the same operational properties MCAP is used for:
append-only chunked records, per-record checksums so a truncated tail
doesn't corrupt prior chunks, bounded file rotation, and offset-addressable
reads. `McapLiteWriter`/`McapLiteReader` implement the `RawExperienceLog`
interface — swap in a real `mcap`-backed implementation behind the same
interface with zero changes to callers (episodic store only ever holds
`(file_id, offset)` provenance pointers, never raw bytes).
"""

from __future__ import annotations

import json
import os
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RawRecord:
    file_id: str
    offset: int
    topic: str
    payload: dict


class RawExperienceLog(ABC):
    @abstractmethod
    def append(self, topic: str, payload: dict) -> tuple[str, int]:
        """Returns (file_id, offset)."""

    @abstractmethod
    def read_range(self, file_id: str, start: int, end: int) -> list[RawRecord]: ...

    @abstractmethod
    def current_file_id(self) -> str: ...


class McapLiteWriter(RawExperienceLog):
    def __init__(self, base_dir: str, max_records_per_file: int = 5000, max_files: int = 20) -> None:
        self.base_dir = base_dir
        self.max_records_per_file = max_records_per_file
        self.max_files = max_files
        os.makedirs(base_dir, exist_ok=True)
        self._file_index = self._discover_next_index()
        self._records_in_current_file = self._count_lines(self._path_for(self._file_index))
        self._enforce_rotation_limit()

    def _path_for(self, index: int) -> str:
        return os.path.join(self.base_dir, f"chunk_{index:06d}.mcaplite")

    def _discover_next_index(self) -> int:
        existing = [f for f in os.listdir(self.base_dir) if f.startswith("chunk_") and f.endswith(".mcaplite")]
        if not existing:
            return 0
        return max(int(f[len("chunk_"):-len(".mcaplite")]) for f in existing)

    def _count_lines(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, "r") as fh:
            return sum(1 for _ in fh)

    def current_file_id(self) -> str:
        return f"chunk_{self._file_index:06d}"

    def append(self, topic: str, payload: dict) -> tuple[str, int]:
        if self._records_in_current_file >= self.max_records_per_file:
            self._file_index += 1
            self._records_in_current_file = 0
            self._enforce_rotation_limit()
        path = self._path_for(self._file_index)
        record = {"topic": topic, "payload": payload}
        body = json.dumps(record, separators=(",", ":"))
        checksum = zlib.crc32(body.encode()) & 0xFFFFFFFF
        line = f"{checksum:08x}|{body}\n"
        with open(path, "a") as fh:
            fh.write(line)
        offset = self._records_in_current_file
        self._records_in_current_file += 1
        return self.current_file_id(), offset

    def _enforce_rotation_limit(self) -> None:
        files = sorted(f for f in os.listdir(self.base_dir) if f.startswith("chunk_") and f.endswith(".mcaplite"))
        while len(files) > self.max_files:
            os.remove(os.path.join(self.base_dir, files.pop(0)))

    def read_range(self, file_id: str, start: int, end: int) -> list[RawRecord]:
        path = os.path.join(self.base_dir, f"{file_id}.mcaplite")
        if not os.path.exists(path):
            return []
        records: list[RawRecord] = []
        with open(path, "r") as fh:
            for i, line in enumerate(fh):
                if i < start:
                    continue
                if i > end:
                    break
                try:
                    checksum_hex, body = line.rstrip("\n").split("|", 1)
                    if zlib.crc32(body.encode()) & 0xFFFFFFFF != int(checksum_hex, 16):
                        break  # corrupted tail — stop here, keep everything read so far
                    parsed = json.loads(body)
                except (ValueError, json.JSONDecodeError):
                    break
                records.append(RawRecord(file_id=file_id, offset=i, topic=parsed["topic"], payload=parsed["payload"]))
        return records


McapLiteReader = McapLiteWriter  # same class supports both; kept as an alias for read-only call sites
