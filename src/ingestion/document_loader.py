"""Document loader — walks data/raw and loads all supported file types.

Supported formats:
    .pdf   — loaded with PyPDFLoader  (page-level metadata)
    .docx  — loaded with Docx2txtLoader
    .doc   — loaded with Docx2txtLoader (requires python-docx)
    .txt   — loaded with TextLoader (UTF-8, falls back to latin-1)

Each document is split into overlapping chunks using
RecursiveCharacterTextSplitter with settings from config.py.
"""

import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


# Map file extension → loader factory
_LOADERS = {
    ".pdf":  lambda p: PyPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".doc":  lambda p: Docx2txtLoader(p),
    ".txt":  lambda p: TextLoader(p, encoding="utf-8"),
}


def _load_file(path: str) -> List[Document]:
    """Load a single file, returning an empty list on failure."""
    ext = os.path.splitext(path)[1].lower()
    factory = _LOADERS.get(ext)
    if factory is None:
        return []
    try:
        return factory(path).load()
    except Exception as exc:
        # Try latin-1 fallback for .txt files
        if ext == ".txt":
            try:
                return TextLoader(path, encoding="latin-1").load()
            except Exception:
                pass
        print(f"[loader] WARNING: skipping {path} — {exc}")
        return []


def load_documents(data_dir: str = "./data/raw") -> List[Document]:
    """Walk *data_dir* recursively, load every supported file, and split into chunks.

    Args:
        data_dir: root directory to walk (e.g. ``./data/raw``).

    Returns:
        List of chunked ``Document`` objects ready to embed and store.
    """
    raw_docs: List[Document] = []
    loaded_files = []

    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):                       # sorted for determinism
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _LOADERS:
                continue
            full_path = os.path.join(root, fname)
            docs = _load_file(full_path)
            if docs:
                raw_docs.extend(docs)
                loaded_files.append(fname)

    if loaded_files:
        print(f"[loader] Loaded {len(loaded_files)} file(s): {', '.join(loaded_files)}")
        print(f"[loader] Total pages/sections before splitting: {len(raw_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[loader] Produced {len(chunks)} chunks.")
    return chunks
