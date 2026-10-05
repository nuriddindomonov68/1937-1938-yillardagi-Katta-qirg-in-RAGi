"""Markaziy sozlamalar. Yo'llarni muhit o'zgaruvchilari orqali o'zgartirish mumkin."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PAGES_JSON = Path(os.getenv("RAG_PAGES_JSON", ROOT / "data" / "parsed_pages.json"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "index"))
DB_PATH = INDEX_DIR / "rag_index.sqlite"
VEC_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 700))      # belgilar soni
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 150))  # bo'laklar orasidagi qoplama
