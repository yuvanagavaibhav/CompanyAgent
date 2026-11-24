import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / ".faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5