import threading
from config import WHISPER_MODEL

_model = None
_model_lock = threading.Lock()


def get_whisper_model():
    """Thread-safe lazy loader for the Whisper model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper  # noqa: PLC0415 – intentional lazy import
                _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_audio(audio_path):
    result = get_whisper_model().transcribe(audio_path)
    return result["text"]
