import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Local Ollama configs
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5-coder:7b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 5

# Ensure directories exist upon startup
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)