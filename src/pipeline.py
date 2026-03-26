from typing import Tuple, Dict, Any, Callable, Optional
from src.ingestion.youtube import fetch_youtube_transcript, download_audio
from src.ingestion.transcribe import transcribe_audio
from src.processing.summarize import summarize_text

def process_youtube_pipeline(url: str, detail_level: str, status_cb: Optional[Callable[[str], None]] = None) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Facade for the entire YouTube processing pipeline.
    Handles extraction, fallback transcription, and initial summarization.
    """
    if status_cb: status_cb("Extracting transcript from YouTube...")
    text, source_type = fetch_youtube_transcript(url)

    if text is not None:
        source = "YouTube Captions (Manual)" if source_type == "manual" else "YouTube Captions (Auto-generated)"
    else:
        if status_cb: status_cb("No captions found. Downloading audio...")
        audio_path = download_audio(url)
        
        if not audio_path:
            raise ValueError("Failed to download audio. Please verify the URL.")
            
        if status_cb: status_cb("Transcribing audio with Whisper (this may take several minutes)...")
        text = transcribe_audio(audio_path)
        source = "Whisper Transcription"

    if status_cb: status_cb("Generating summary with BART-large-CNN...")
    summary, metrics = summarize_text(
        text, 
        detail_level=detail_level,
        model_name="bart-large-cnn",
        return_metrics=True
    )
    
    return text, source, summary, metrics

def process_audio_pipeline(file_path: str, detail_level: str, status_cb: Optional[Callable[[str], None]] = None) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Facade for processing raw audio files.
    """
    if status_cb: status_cb("Transcribing audio with Whisper (this may take several minutes)...")
    text = transcribe_audio(file_path)
    source = "Whisper Transcription"

    if status_cb: status_cb("Generating summary with BART-large-CNN...")
    summary, metrics = summarize_text(
        text,
        detail_level=detail_level,
        model_name="bart-large-cnn",
        return_metrics=True
    )

    return text, source, summary, metrics
