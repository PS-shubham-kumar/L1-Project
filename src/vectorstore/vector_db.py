from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DB_PATH, COLLECTION_NAME


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
        )

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
