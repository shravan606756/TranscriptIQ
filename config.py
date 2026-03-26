import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file (e.g., GROQ_API_KEY, PYTHONPATH)
load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Matches the 'data/audio' path used in app/app.py and src/ingestion/youtube.py
AUDIO_CACHE_DIR = os.getenv("AUDIO_CACHE_DIR", str(DATA_DIR / "audio"))

# --- API Keys ---
# Used natively in src/retrieval/rag.py
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

# --- Model Configurations ---
# Transcription (used in src/ingestion/transcribe.py)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Hardware/Device settings
DEVICE = os.getenv("DEVICE", "cpu")  # Defaults to CPU since summarize.py uses device=-1

# RAG & Embeddings (used in src/retrieval/rag.py)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 4))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 350))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 60))

# Summarization Limits & Chunker Boundaries (as requested by README)
BART_MAX_INPUT_TOKENS = int(os.getenv("BART_MAX_INPUT_TOKENS", 1024))
T5_MAX_INPUT_TOKENS = int(os.getenv("T5_MAX_INPUT_TOKENS", 512))
BART_MAX_NEW_TOKENS = int(os.getenv("BART_MAX_NEW_TOKENS", 150))
T5_MAX_NEW_TOKENS = int(os.getenv("T5_MAX_NEW_TOKENS", 100))

# Create directories if they don't exist
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_CACHE_DIR = os.path.join(DATA_DIR, "audio")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Models Configuration
WHISPER_MODEL = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEVICE = os.getenv("DEVICE", "cpu")

# RAG & Chunking Hyperparameters
RAG_TOP_K = 4
RAG_CHUNK_SIZE = 350
RAG_CHUNK_OVERLAP = 60

# Summarizer Limits
BART_MAX_INPUT_TOKENS = 1024
T5_MAX_INPUT_TOKENS = 512
