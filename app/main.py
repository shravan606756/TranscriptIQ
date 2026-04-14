"""
TranscriptIQ – FastAPI backend
==============================

Endpoints
---------
POST /api/v1/ingest/youtube        Enqueue a YouTube processing job
GET  /api/v1/job/{job_id}          Poll job status
GET  /api/v1/summary/{video_id}    Retrieve a stored summary
GET  /api/v1/compare/{video_id}    Compare both model summaries
GET  /health                        Liveness check
"""

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.jobs import create_job, get_job, update_job
from src.pipeline import process_youtube_pipeline
from src.processing.summarize import summarize_text
from src.retrieval.rag import build_vector_store
from src.storage import (
    load_summary,
    save_index,
    save_summary,
    save_transcript,
    summary_exists,
)
from src.utils import video_id_from_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TranscriptIQ API",
    description="Backend API for TranscriptIQ – YouTube transcript summarization and RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IngestYouTubeRequest(BaseModel):
    url: str
    detail_level: Literal["brief", "medium", "detailed"] = "medium"
    model: Literal["bart", "t5", "both"] = "bart"

    @field_validator("url")
    @classmethod
    def url_must_be_youtube(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        # Delegate deep validation to video_id_from_url; raises ValueError on bad input.
        try:
            video_id_from_url(v)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return v


class JobResponse(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------

_MODEL_KEY_MAP = {
    "bart": "bart-large-cnn",
    "t5": "t5-base",
}


def run_youtube_job(job_id: str, url: str, detail_level: str, model: str) -> None:
    """
    Heavy lifting executed inside FastAPI BackgroundTasks.

    Steps
    -----
    1. Extract / transcribe the YouTube video.
    2. Save the transcript to disk.
    3. Summarise with BART (always) and optionally with T5.
    4. Build and persist the FAISS index.
    5. Update the job record to "done" (or "error" on failure).
    """
    try:
        video_id = video_id_from_url(url)
    except ValueError as exc:
        update_job(job_id, status="error", error=str(exc))
        return

    try:
        # --- Step 1: transcript ------------------------------------------------
        update_job(job_id, status="running", progress="Extracting transcript…")
        text, source, bart_summary, bart_metrics = process_youtube_pipeline(
            url, detail_level
        )

        # --- Step 2: persist transcript ----------------------------------------
        update_job(job_id, progress="Saving transcript…")
        save_transcript(video_id, text, meta={"source": source, "url": url})

        # --- Step 3: BART summary (returned by pipeline) ----------------------
        update_job(job_id, progress="Saving BART summary…")
        save_summary(video_id, "bart-large-cnn", bart_summary, bart_metrics)

        # --- Step 3b: T5 summary (on request) ---------------------------------
        requested_model_key = _MODEL_KEY_MAP.get(model, "bart-large-cnn")
        run_t5 = model in ("t5", "both")
        if run_t5 and not summary_exists(video_id, "t5-base"):
            update_job(job_id, progress="Generating T5 summary…")
            t5_summary, t5_metrics = summarize_text(
                text, detail_level=detail_level, model_name="t5-base", return_metrics=True
            )
            save_summary(video_id, "t5-base", t5_summary, t5_metrics)

        # --- Step 4: FAISS index -----------------------------------------------
        update_job(job_id, progress="Building vector index…")
        index, chunks = build_vector_store(text)
        if index is not None:
            save_index(video_id, index, chunks)

        # --- Step 5: done -------------------------------------------------------
        result = {
            "video_id": video_id,
            "source": source,
            "models_run": ["bart-large-cnn"] + (["t5-base"] if run_t5 else []),
        }
        update_job(job_id, status="done", progress="Completed", result=result)

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        update_job(
            job_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.post("/api/v1/ingest/youtube", status_code=202, response_model=JobResponse, tags=["ingest"])
def ingest_youtube(
    body: IngestYouTubeRequest,
    background_tasks: BackgroundTasks,
):
    """Enqueue a YouTube video for transcription and summarisation."""
    job = create_job()
    background_tasks.add_task(
        run_youtube_job,
        job["job_id"],
        body.url,
        body.detail_level,
        body.model,
    )
    return {"job_id": job["job_id"]}


@app.get("/api/v1/job/{job_id}", tags=["jobs"])
def get_job_status(job_id: str):
    """Return the current status of a background job."""
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return record


@app.get("/api/v1/summary/{video_id}", tags=["summaries"])
def get_summary(video_id: str, model: str = "bart"):
    """
    Return the stored summary for *video_id*.

    Query params
    ------------
    model : "bart" (default) | "t5"
    """
    model_key = _MODEL_KEY_MAP.get(model, "bart-large-cnn")
    data = load_summary(video_id, model_key)
    if data is None:
        raise HTTPException(status_code=404, detail="Summary not found")

    other_key = "t5-base" if model_key == "bart-large-cnn" else "bart-large-cnn"
    comparison_available = summary_exists(video_id, other_key)

    return {
        **data,
        "video_id": video_id,
        "comparison_available": comparison_available,
    }


@app.get("/api/v1/compare/{video_id}", tags=["summaries"])
def compare_summaries(video_id: str):
    """
    Return both BART and T5 summaries together with lightweight comparison metrics.

    Raises 404 if both summaries are not yet available.
    """
    bart_data = load_summary(video_id, "bart-large-cnn")
    t5_data = load_summary(video_id, "t5-base")

    if bart_data is None or t5_data is None:
        missing = []
        if bart_data is None:
            missing.append("bart-large-cnn")
        if t5_data is None:
            missing.append("t5-base")
        raise HTTPException(
            status_code=404,
            detail=f"Summaries not yet available for model(s): {missing}",
        )

    # Lightweight comparison metrics (word counts, compression ratios)
    def _wc(text: str) -> int:
        return len(text.split())

    bart_words = _wc(bart_data["summary"])
    t5_words = _wc(t5_data["summary"])

    comparison = {
        "bart_word_count": bart_words,
        "t5_word_count": t5_words,
        "bart_compression_ratio": bart_data.get("metrics", {}).get("compression_ratio"),
        "t5_compression_ratio": t5_data.get("metrics", {}).get("compression_ratio"),
        "bart_processing_time": bart_data.get("metrics", {}).get("processing_time"),
        "t5_processing_time": t5_data.get("metrics", {}).get("processing_time"),
    }

    return {
        "video_id": video_id,
        "bart": bart_data,
        "t5": t5_data,
        "comparison": comparison,
    }
