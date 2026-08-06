"""
File-backed job store for FastAPI BackgroundTasks.

Jobs are persisted to  data/jobs/<job_id>.json  so their status survives
a worker restart (within the same run).  Each record is a plain dict:

{
    "job_id":    str,
    "status":    "pending" | "running" | "done" | "error",
    "progress":  str,          # human-readable status message
    "result":    dict | None,  # set when status == "done"
    "error":     str | None,   # set when status == "error"
    "created_at": ISO-8601 str,
    "updated_at": ISO-8601 str,
}
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import BASE_DIR

_JOBS_DIR = Path(BASE_DIR) / "data" / "jobs"

# Job IDs are always UUID4 strings – allow only that pattern.
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _validate_job_id(job_id: str) -> None:
    """Raise ValueError if *job_id* is not a valid UUID4 string."""
    if not _JOB_ID_RE.match(job_id):
        raise ValueError(f"Invalid job_id format: {job_id!r}")


def _jobs_dir() -> Path:
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return _JOBS_DIR


def _job_path(job_id: str) -> Path:
    _validate_job_id(job_id)
    return _jobs_dir() / f"{job_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job() -> Dict[str, Any]:
    """Create a new job record, persist it, and return the record."""
    job_id = str(uuid.uuid4())
    record: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "progress": "Queued",
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _save(record)
    return record


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Update fields on an existing job record and persist the change."""
    record = get_job(job_id)
    if record is None:
        raise ValueError(f"Job {job_id!r} not found")
    if status is not None:
        record["status"] = status
    if progress is not None:
        record["progress"] = progress
    if result is not None:
        record["result"] = result
    if error is not None:
        record["error"] = error
    record["updated_at"] = _now_iso()
    _save(record)
    return record


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job record or None if it does not exist."""
    try:
        path = _job_path(job_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(record: Dict[str, Any]) -> None:
    path = _job_path(record["job_id"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
