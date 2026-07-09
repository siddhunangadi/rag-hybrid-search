"""In-memory background job tracking for async document ingestion.

Ingesting a document (parse -> chunk -> embed -> index) can take minutes for
large files, which would otherwise block the request thread and time out the
client (see IndexResult-based synchronous /upload for the blocking version).
JobStore lets a route hand the work to a single-worker background thread and
return a job_id immediately; the client polls GET /jobs/{job_id} for status.

Single-process, in-memory only: state is lost on restart and isn't shared
across multiple API instances. That's consistent with this project's current
single-instance deployment model (see other known scaling limitations); a
production multi-instance deployment would need a shared job store (e.g.
Redis) instead.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal, Optional

JobState = Literal["processing", "ready", "failed"]


@dataclass
class Job:
    job_id: str
    state: JobState = "processing"
    result: Optional[dict] = None
    error: Optional[str] = None


class JobStore:
    """Tracks background ingestion jobs, executed one at a time on a dedicated worker thread.

    A single worker (``max_workers=1``) deliberately serializes ingestion so
    concurrent uploads can't race on the shared BM25 rebuild or chunk store
    writes -- the same safety property the old synchronous endpoint got for
    free by running on one thread.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingestion-worker")

    def submit(self, work: Callable[[], dict]) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = Job(job_id=job_id)
        self._executor.submit(self._run, job_id, work)
        return job_id

    def _run(self, job_id: str, work: Callable[[], dict]) -> None:
        try:
            result = work()
            with self._lock:
                self._jobs[job_id] = Job(job_id=job_id, state="ready", result=result)
        except Exception as e:  # noqa: BLE001 - surface any failure via job status, not a crash
            with self._lock:
                self._jobs[job_id] = Job(job_id=job_id, state="failed", error=str(e))

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
