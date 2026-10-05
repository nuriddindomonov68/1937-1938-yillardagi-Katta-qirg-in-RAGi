"""Matnni normallashtirish va bo'laklarga (chunk) bo'lish."""
import re
import unicodedata

# O'zbek lotin alifbosidagi turli apostrof belgilarini bitta ko'rinishga keltiramiz
_APOSTROPHES = "\u02bb\u02bc\u2018\u2019\u0060\u00b4\u2032"
_APOS_RE = re.compile(f"[{_APOSTROPHES}]")


def normalize(text: str) -> str:
    """Unicode NFKC, apostroflarni birxillashtirish va ortiqcha bo'shliqlarni olib tashlash.

    Misol: ``o‘zbek``, ``oʻzbek``, ``o`zbek`` -> ``o'zbek``.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _APOS_RE.sub("'", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


# Apostrofni so'z ichida saqlaydigan tokenizer (o'zbek, g'oya, ma'lumot)
TOKEN_PATTERN = r"(?u)\b\w+(?:'\w+)*\b"


def split_into_chunks(text: str, page_no: int, chunk_size: int = 700, overlap: int = 150):
    """Sahifa matnini so'z chegarasiga moslab, qoplamali bo'laklarga bo'ladi."""
    if chunk_size <= 0:
        raise ValueError("chunk_size musbat bo'lishi kerak")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 0 va chunk_size orasida bo'lishi kerak")

    text = re.sub(r"\n{2,}", "\n", text).strip()
    if not text:
        return []

    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            chunks.append({"page": page_no, "text": piece})
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def preprocess(text: str) -> str:
    """TfidfVectorizer uchun preprocessor (pickle qilinishi uchun modul darajasida)."""
    return normalize(text).lower()
