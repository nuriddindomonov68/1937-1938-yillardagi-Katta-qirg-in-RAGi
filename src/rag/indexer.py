"""Indeks qurish: sahifalar -> bo'laklar -> SQLite + TF-IDF."""
import json
import pickle
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .text_utils import TOKEN_PATTERN, preprocess, split_into_chunks


def load_pages(path: Path = config.PAGES_JSON):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} topilmadi. Avval PDF ni parse qiling: `python -m rag parse <fayl.pdf>`"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(
    pages_json: Path = config.PAGES_JSON,
    index_dir: Path = config.INDEX_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> int:
    """Indeksni quradi va diskka saqlaydi. Bo'laklar sonini qaytaradi."""
    pages = load_pages(pages_json)
    chunks = []
    for p in pages:
        chunks.extend(split_into_chunks(p["text"], p["page"], chunk_size, overlap))
    if not chunks:
        raise ValueError("Hech qanday matn bo'lagi topilmadi - kiruvchi fayl bo'sh.")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / config.DB_PATH.name
    vec_path = index_dir / config.VEC_PATH.name

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, page INTEGER, text TEXT)")
        cur.executemany(
            "INSERT INTO chunks (id, page, text) VALUES (?, ?, ?)",
            [(i, c["page"], c["text"]) for i, c in enumerate(chunks)],
        )
        conn.commit()
    finally:
        conn.close()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=TOKEN_PATTERN,
        preprocessor=preprocess,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    with open(vec_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    print(f"Sahifalar: {len(pages)}, bo'laklar: {len(chunks)}")
    print(f"Indeks saqlandi: {index_dir}")
    return len(chunks)
