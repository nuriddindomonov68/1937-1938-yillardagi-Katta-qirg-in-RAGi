"""Lokal, offline RAG (Retrieval-Augmented Generation) tizimi - o'zbek tili uchun."""
from .retriever import Retriever, answer, retrieve
from .indexer import build_index

__all__ = ["Retriever", "answer", "retrieve", "build_index"]
__version__ = "0.2.0"
