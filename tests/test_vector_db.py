"""Tests for vector store creation and configuration.

Uses mocked Chroma to verify that ``get_vectorstore`` correctly routes
between creating a new store (when docs are provided) and opening an
existing one (when they are not), and that config paths are applied.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from config import CHROMA_DB_PATH, COLLECTION_NAME
from vectorstore.vector_db import _COLLECTION_METADATA


class TestGetVectorstore:
    """Verify get_vectorstore routes correctly based on whether docs are provided."""

    @patch("vectorstore.vector_db.Chroma")
    def test_opens_existing_store_when_no_docs(self, MockChroma):
        from vectorstore.vector_db import get_vectorstore

        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        MockChroma.return_value = mock_store

        result = get_vectorstore(mock_embeddings)

        MockChroma.assert_called_once_with(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=mock_embeddings,
            collection_name=COLLECTION_NAME,
            collection_metadata=_COLLECTION_METADATA,
        )
        assert result is mock_store

    @patch("vectorstore.vector_db.Chroma")
    def test_creates_store_from_documents(self, MockChroma):
        from vectorstore.vector_db import get_vectorstore

        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        MockChroma.from_documents.return_value = mock_store

        docs = [Document(page_content="chunk 1", metadata={"source": "test.txt"})]
        result = get_vectorstore(mock_embeddings, docs)

        MockChroma.from_documents.assert_called_once_with(
            documents=docs,
            embedding=mock_embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            collection_metadata=_COLLECTION_METADATA,
        )
        assert result is mock_store

    @patch("vectorstore.vector_db.Chroma")
    def test_empty_doc_list_opens_existing_store(self, MockChroma):
        """An empty list is falsy, so it should open the existing store."""
        from vectorstore.vector_db import get_vectorstore

        mock_embeddings = MagicMock()
        get_vectorstore(mock_embeddings, docs=[])

        MockChroma.from_documents.assert_not_called()
        MockChroma.assert_called_once()

    @patch("vectorstore.vector_db.Chroma")
    def test_store_uses_configured_paths(self, MockChroma):
        """Verify the store uses CHROMA_DB_PATH and COLLECTION_NAME from config."""
        from vectorstore.vector_db import get_vectorstore

        mock_embeddings = MagicMock()
        get_vectorstore(mock_embeddings)

        call_kwargs = MockChroma.call_args[1]
        assert call_kwargs["persist_directory"] == CHROMA_DB_PATH
        assert call_kwargs["collection_name"] == COLLECTION_NAME
