"""
Persistent storage helpers for transcripts, summaries, and FAISS indexes.

Directory layout under  data/  (BASE_DIR/data):

    data/
    ├── jobs/               ← job status files (managed by src/jobs.py)
    ├── indexes/
    │   ├── <video_id>.index        ← FAISS binary index
    │   └── <video_id>_chunks.json  ← serialised chunk list
    └── <video_id>/
        ├── transcript.json.gz
        └── summary_<model_key>.json.gz

All transcript and summary files are gzip-compressed JSON to reduce disk footprint.
"""

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss

from config import BASE_DIR

_DATA_DIR = Path(BASE_DIR) / "data"
_INDEX_DIR = _DATA_DIR / "indexes"

# YouTube video IDs are 11 characters of [A-Za-z0-9_-].
# We allow the same character set but do not enforce length strictly so that
# future ID formats keep working.  Path-traversal characters are forbidden.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_id(value: str, label: str = "id") -> None:
    """Raise ValueError if *value* contains unsafe path characters."""
    if not _SAFE_ID_RE.match(value):
        raise ValueError(f"Unsafe {label}: {value!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _video_dir(video_id: str) -> Path:
    _validate_id(video_id, "video_id")
    d = _DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_dir() -> Path:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return _INDEX_DIR


def _gz_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _gz_read(path: Path) -> Any:
    with gzip.open(str(path), "rt", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def save_transcript(video_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist transcript text (and optional metadata) for *video_id*."""
    payload = {"text": text, "meta": meta or {}}
    path = _video_dir(video_id) / "transcript.json.gz"
    _gz_write(path, payload)


def load_transcript(video_id: str) -> Optional[Dict[str, Any]]:
    """Return ``{"text": ..., "meta": ...}`` or *None* if not found."""
    _validate_id(video_id, "video_id")
    path = _DATA_DIR / video_id / "transcript.json.gz"
    if not path.exists():
        return None
    return _gz_read(path)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

# Allow only known model keys to prevent path traversal via model_key parameter.
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _validate_model_key(model_key: str) -> None:
    if not _SAFE_MODEL_RE.match(model_key):
        raise ValueError(f"Unsafe model_key: {model_key!r}")


def _summary_path(video_id: str, model_key: str) -> Path:
    _validate_id(video_id, "video_id")
    _validate_model_key(model_key)
    safe_key = model_key.replace("/", "_").replace("\\", "_")
    return _DATA_DIR / video_id / f"summary_{safe_key}.json.gz"


def save_summary(
    video_id: str,
    model_key: str,
    summary_text: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a summary produced by *model_key* for *video_id*."""
    payload = {"summary": summary_text, "metrics": metrics or {}, "model_key": model_key}
    path = _summary_path(video_id, model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    _gz_write(path, payload)


def load_summary(video_id: str, model_key: str) -> Optional[Dict[str, Any]]:
    """Return ``{"summary": ..., "metrics": ..., "model_key": ...}`` or *None*."""
    path = _summary_path(video_id, model_key)
    if not path.exists():
        return None
    return _gz_read(path)


def summary_exists(video_id: str, model_key: str) -> bool:
    """Return *True* if a summary for this (video, model) pair has been persisted."""
    try:
        return _summary_path(video_id, model_key).exists()
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# FAISS index + chunks
# ---------------------------------------------------------------------------

def save_index(video_id: str, index: Any, chunks: List[str]) -> None:
    """Write the FAISS index and corresponding chunk list to disk."""
    _validate_id(video_id, "video_id")
    index_path = str(_index_dir() / f"{video_id}.index")
    faiss.write_index(index, index_path)

    chunks_path = _index_dir() / f"{video_id}_chunks.json"
    with open(str(chunks_path), "w", encoding="utf-8") as fh:
        json.dump(chunks, fh)


def load_index_and_chunks(video_id: str) -> Tuple[Optional[Any], List[str]]:
    """
    Load the FAISS index and chunk list for *video_id*.

    Returns ``(index, chunks)`` where either may be *None* / empty list if not found.
    """
    _validate_id(video_id, "video_id")
    index_path = _index_dir() / f"{video_id}.index"
    chunks_path = _index_dir() / f"{video_id}_chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))
    with open(str(chunks_path), "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    return index, chunks
