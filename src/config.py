import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Service & LLM Provider Configs (loaded from environment with fallbacks)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:12b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Optional External Providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Embeddings Config
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# RAG Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 5))

# Execution & Resilience Settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 15))

# Ensure directories exist upon startup
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)