"""Qidiruv (retrieval) va LLM uchun prompt tayyorlash."""
import pickle
import sqlite3
from functools import lru_cache
from pathlib import Path

from sklearn.metrics.pairwise import linear_kernel

from . import config


class Retriever:
    """Indeksni bir marta yuklab, ko'p so'rovlarga xizmat qiladi."""

    def __init__(self, index_dir: Path = config.INDEX_DIR):
        index_dir = Path(index_dir)
        db_path = index_dir / config.DB_PATH.name
        vec_path = index_dir / config.VEC_PATH.name
        if not (db_path.exists() and vec_path.exists()):
            raise FileNotFoundError(
                f"Indeks topilmadi ({index_dir}). Avval `python -m rag build` ni ishga tushiring."
            )
        # Diqqat: pickle faqat o'zingiz qurgan, ishonchli fayllar uchun ishlating.
        with open(vec_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        conn = sqlite3.connect(db_path)
        try:
            self.rows = conn.execute("SELECT page, text FROM chunks ORDER BY id").fetchall()
        finally:
            conn.close()

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.0):
        qvec = self.vectorizer.transform([query])
        sims = linear_kernel(qvec, self.matrix)[0]  # TF-IDF L2-normallashgan: cosine bilan teng
        top = sims.argsort()[::-1][:k]
        return [
            {"page": self.rows[i][0], "text": self.rows[i][1], "score": float(sims[i])}
            for i in top
            if sims[i] > min_score
        ]

    def answer(self, query: str, k: int = 5, min_score: float = 0.0):
        results = self.retrieve(query, k, min_score)
        context = "\n\n---\n\n".join(
            f"[Sahifa {r['page']}, skor={r['score']:.3f}]\n{r['text']}" for r in results
        )
        prompt = (
            "Quyidagi manbalar asosida savolga javob bering. "
            "Faqat berilgan kontekstdagi ma'lumotlardan foydalaning, sahifa raqamlarini keltiring. "
            "Agar javob kontekstda bo'lmasa, buni ochiq ayting.\n\n"
            f"SAVOL: {query}\n\n"
            f"KONTEKST:\n{context}\n"
        )
        return {"query": query, "retrieved": results, "prompt": prompt}


@lru_cache(maxsize=1)
def _default() -> Retriever:
    return Retriever()


def retrieve(query: str, k: int = 5):
    return _default().retrieve(query, k)


def answer(query: str, k: int = 5):
    return _default().answer(query, k)
