from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DB_PATH, COLLECTION_NAME

# Use cosine distance so that relevance scores equal cosine similarity (0–1).
# The default L2 distance produces scores via  1 − dist/√2  which are much
# lower and make the SIMILARITY_THRESHOLD in config.py hard to reason about.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_vectorstore(embeddings, docs: Optional[List[Document]] = None) -> Chroma:
    """Return a Chroma vectorstore.

    If *docs* are provided, (re)build the store from those documents.
    Otherwise, open the existing persisted store for querying.
    """
    if docs:
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            collection_metadata=_COLLECTION_METADATA,
        )

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata=_COLLECTION_METADATA,
    )

