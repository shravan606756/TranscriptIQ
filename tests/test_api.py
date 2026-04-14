"""
Minimal FastAPI endpoint tests.

Heavy ML dependencies (transformers, torch, faiss, whisper, sentence_transformers)
are stubbed via sys.modules before any project code is imported, so the suite
runs without installing or loading any ML model.
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing project modules
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> ModuleType:
    mod = ModuleType(name)
    return mod


_HEAVY = [
    "whisper",
    "torch",
    "faiss",
    "transformers",
    "sentence_transformers",
    "openai",
    "gtts",
]

for _stub in _HEAVY:
    if _stub not in sys.modules:
        sys.modules[_stub] = _make_stub(_stub)

# faiss needs write_index / read_index / IndexFlatL2
_faiss = sys.modules["faiss"]
_faiss.write_index = MagicMock()
_faiss.read_index = MagicMock(return_value=MagicMock())
_faiss.IndexFlatL2 = MagicMock()

# transformers needs pipeline
sys.modules["transformers"].pipeline = MagicMock()

# sentence_transformers needs SentenceTransformer
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()

# openai needs OpenAI class
sys.modules["openai"].OpenAI = MagicMock()

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/v1/ingest/youtube
# ---------------------------------------------------------------------------


def test_ingest_youtube_returns_job_id(tmp_path, monkeypatch):
    """Posting a valid YouTube URL should enqueue a job and return a job_id."""
    # Redirect data/jobs to a temp directory so tests don't pollute the repo
    monkeypatch.setattr("src.jobs._JOBS_DIR", tmp_path / "jobs")

    # Mock the heavy pipeline functions so the background task completes fast
    with patch("app.main.process_youtube_pipeline") as mock_pipeline, \
         patch("app.main.build_vector_store") as mock_vs, \
         patch("app.main.save_transcript"), \
         patch("app.main.save_summary"), \
         patch("app.main.save_index"), \
         patch("app.main.summary_exists", return_value=False):

        mock_pipeline.return_value = (
            "Transcript text",
            "YouTube Captions",
            "BART summary",
            {"compression_ratio": 50.0, "processing_time": 1.0},
        )
        mock_vs.return_value = (MagicMock(), ["chunk1", "chunk2"])

        response = client.post(
            "/api/v1/ingest/youtube",
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "detail_level": "medium",
                "model": "bart",
            },
        )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert isinstance(data["job_id"], str)
    assert len(data["job_id"]) > 0


# ---------------------------------------------------------------------------
# GET /api/v1/job/{job_id}
# ---------------------------------------------------------------------------


def test_get_job_status_not_found():
    response = client.get("/api/v1/job/nonexistent-job-id")
    assert response.status_code == 404


def test_get_job_status_found(tmp_path, monkeypatch):
    monkeypatch.setattr("src.jobs._JOBS_DIR", tmp_path / "jobs")

    from src.jobs import create_job, get_job

    job = create_job()
    job_id = job["job_id"]

    response = client.get(f"/api/v1/job/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# GET /api/v1/summary/{video_id}
# ---------------------------------------------------------------------------


def test_get_summary_not_found():
    with patch("app.main.load_summary", return_value=None):
        response = client.get("/api/v1/summary/some_video_id?model=bart")
    assert response.status_code == 404


def test_get_summary_found():
    fake_summary = {
        "summary": "This is a test summary.",
        "metrics": {"compression_ratio": 60.0},
        "model_key": "bart-large-cnn",
    }
    # BART summary exists; T5 summary does not yet exist → comparison_available=False
    with patch("app.main.load_summary", return_value=fake_summary), \
         patch("app.main.summary_exists", return_value=False):
        response = client.get("/api/v1/summary/test_video?model=bart")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "This is a test summary."
    # T5 summary not present so comparison is not available
    assert data["comparison_available"] is False
    assert data["video_id"] == "test_video"


def test_get_summary_comparison_available():
    """When the other model's summary also exists, comparison_available should be True."""
    fake_summary = {
        "summary": "This is a test summary.",
        "metrics": {"compression_ratio": 60.0},
        "model_key": "bart-large-cnn",
    }
    # Both BART (returned by load_summary) and T5 (summary_exists=True) are present
    with patch("app.main.load_summary", return_value=fake_summary), \
         patch("app.main.summary_exists", return_value=True):
        response = client.get("/api/v1/summary/test_video?model=bart")

    assert response.status_code == 200
    data = response.json()
    assert data["comparison_available"] is True


# ---------------------------------------------------------------------------
# GET /api/v1/compare/{video_id}
# ---------------------------------------------------------------------------


def test_compare_missing_one_model():
    with patch("app.main.load_summary", side_effect=[None, None]):
        response = client.get("/api/v1/compare/test_video")
    assert response.status_code == 404


def test_compare_both_available():
    bart = {
        "summary": "BART summary text.",
        "metrics": {"compression_ratio": 55.0, "processing_time": 2.0},
        "model_key": "bart-large-cnn",
    }
    t5 = {
        "summary": "T5 summary text.",
        "metrics": {"compression_ratio": 60.0, "processing_time": 1.5},
        "model_key": "t5-base",
    }

    def _load_side_effect(video_id, model_key):
        if model_key == "bart-large-cnn":
            return bart
        if model_key == "t5-base":
            return t5
        return None

    with patch("app.main.load_summary", side_effect=_load_side_effect):
        response = client.get("/api/v1/compare/test_video")

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "test_video"
    assert "bart" in data
    assert "t5" in data
    assert "comparison" in data
    assert data["comparison"]["bart_word_count"] == 3
    assert data["comparison"]["t5_word_count"] == 3
