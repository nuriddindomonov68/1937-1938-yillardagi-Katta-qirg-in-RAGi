import json

import pytest

from rag.indexer import build_index
from rag.retriever import Retriever
from rag.text_utils import normalize, split_into_chunks


def test_normalize_apostrophes():
    assert normalize("o‘zbek oʻzbek o`zbek") == "o'zbek o'zbek o'zbek"


def test_chunking_overlap_and_page():
    text = " ".join(f"so'z{i}" for i in range(300))
    chunks = split_into_chunks(text, 7, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(c["page"] == 7 and len(c["text"]) <= 200 for c in chunks)


def test_chunking_invalid_params():
    with pytest.raises(ValueError):
        split_into_chunks("abc", 1, chunk_size=10, overlap=10)


def test_end_to_end(tmp_path):
    pages = [
        {"page": 1, "text": "Toshkent shahri O'zbekiston poytaxtidir."},
        {"page": 2, "text": "Samarqand qadimiy Ipak yo'li shahri hisoblanadi."},
    ]
    pj = tmp_path / "pages.json"
    pj.write_text(json.dumps(pages), encoding="utf-8")
    idx = tmp_path / "index"
    build_index(pj, idx, chunk_size=200, overlap=20)
    res = Retriever(idx).retrieve("Samarqand haqida", k=1)
    assert res and res[0]["page"] == 2
