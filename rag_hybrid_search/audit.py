"""Lightweight, append-only audit event log.

One flat ``AuditEvent`` schema covers query/upload/deletion/supersession/
auth-failure events instead of a class hierarchy -- keeps this additive and
easy to query. Persisted as JSONL (one event per line, never rewritten) next
to the existing local BM25 index file, avoiding a new dependency or a
separate audit service.

ponytail: ``AuditLog.query`` reads and filters the whole file in memory on
every call -- fine at this project's single-developer/portfolio scale.
Swap for sqlite (with an index on timestamp) if the file grows large enough
for that scan to matter.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

EventType = Literal[
    "query", "upload", "deletion", "supersession", "auth_failure", "admin_action",
]
EventStatus = Literal["success", "failure"]


class AuditEvent(BaseModel):
    event_id: str
    event_type: EventType
    timestamp: datetime
    request_id: str
    key_id: str
    role: str | None = None
    endpoint: str
    action: str
    status: EventStatus
    duration_ms: float | None = None
    error: str | None = None

    # Document-related (upload/deletion/supersession)
    document_id: str | None = None
    regulation_metadata: dict | None = None

    # Query/retrieval-related (query)
    query_text: str | None = None
    metadata_filters: dict | None = None
    retrieval_mode: str | None = None
    retrieved_document_ids: list[str] | None = None
    cited_regulations: list[str] | None = None
    confidence_score: float | None = None
    retrieval_stats: dict | None = None


class AuditLog:
    """Append-only JSONL-backed audit event store."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: AuditEvent) -> None:
        line = event.model_dump_json()
        with self._lock:
            with open(self._path, "a") as f:
                f.write(line + "\n")

    def ping(self) -> None:
        """Cheap availability check for readiness probes: the log directory
        must exist and be writable. Doesn't read the (potentially large)
        log file itself."""
        if not self._path.parent.is_dir():
            raise RuntimeError(f"audit log directory missing: {self._path.parent}")

    def count(self) -> int:
        """Total number of recorded events, for the diagnostics endpoint."""
        return len(self._read_all())

    def _read_all(self) -> list[AuditEvent]:
        with self._lock:
            if not self._path.exists():
                return []
            raw = self._path.read_text()
        return [
            AuditEvent.model_validate_json(line)
            for line in raw.splitlines()
            if line.strip()
        ]

    def query(
        self,
        event_type: EventType | None = None,
        key_id: str | None = None,
        role: str | None = None,
        document_id: str | None = None,
        status: EventStatus | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        sort: Literal["asc", "desc"] = "desc",
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[AuditEvent], int]:
        """Filter, sort, and paginate audit events. Returns (page, total_matching)."""
        events = self._read_all()
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        if key_id is not None:
            events = [e for e in events if e.key_id == key_id]
        if role is not None:
            events = [e for e in events if e.role == role]
        if document_id is not None:
            events = [e for e in events if e.document_id == document_id]
        if status is not None:
            events = [e for e in events if e.status == status]
        if start is not None:
            events = [e for e in events if e.timestamp >= start]
        if end is not None:
            events = [e for e in events if e.timestamp <= end]

        events.sort(key=lambda e: e.timestamp, reverse=(sort == "desc"))
        total = len(events)
        return events[offset : offset + limit], total


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
