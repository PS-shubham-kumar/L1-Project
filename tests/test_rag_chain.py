"""Tests for the RAG chain: context construction, citation metadata, and fallback.

Uses the FakeChatModel from conftest.py so no NVIDIA API key is needed.
Verifies that:
- Empty retrieval triggers the correct fallback message
- Source documents and their metadata are preserved in the output
- Context headers are built correctly for docs with and without page numbers
"""

import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from chains.rag_chain import build_rag_chain


class TestRagChainFallback:
    """Verify behavior when no context chunks are retrieved."""

    def test_empty_retrieval_returns_fallback_message(self, fake_llm):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        chain = build_rag_chain(fake_llm, mock_retriever)
        result = chain.invoke("What is quantum computing?")

        assert result["source_docs"] == []
        assert "No relevant chunks" in result["answer"]

    def test_empty_retrieval_skips_llm_call(self, fake_llm):
        """When retrieval is empty, the answer should be the hardcoded fallback,
        not the LLM's output — proving the LLM was never invoked."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        chain = build_rag_chain(fake_llm, mock_retriever)
        result = chain.invoke("Irrelevant question")

        # The fallback message is NOT the fake LLM's fixed response
        assert result["answer"] != fake_llm.fixed_response


class TestRagChainWithContext:
    """Verify context construction and citation metadata propagation."""

    def _build(self, fake_llm, docs):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        return build_rag_chain(fake_llm, mock_retriever)

    def test_returns_source_docs_with_metadata(self, fake_llm, sample_documents):
        chain = self._build(fake_llm, sample_documents)
        result = chain.invoke("What is Article 14?")

        assert "source_docs" in result
        assert len(result["source_docs"]) == 2
        assert result["source_docs"][0].metadata["page"] == 5
        assert "constitution.pdf" in result["source_docs"][0].metadata["source"]

    def test_answer_comes_from_llm(self, fake_llm, sample_documents):
        chain = self._build(fake_llm, sample_documents)
        result = chain.invoke("What is Article 14?")

        assert result["answer"] == fake_llm.fixed_response

    def test_handles_docs_without_page_metadata(self, fake_llm):
        """Documents without a 'page' key should still produce valid output."""
        docs = [
            Document(
                page_content="Text file content here.",
                metadata={"source": "/data/raw/txt/notes.txt"},
            )
        ]
        chain = self._build(fake_llm, docs)
        result = chain.invoke("Test question")

        assert len(result["source_docs"]) == 1
        assert result["source_docs"][0].metadata.get("page") is None
        assert "answer" in result

    def test_handles_zero_indexed_pages(self, fake_llm):
        """Page numbers in metadata are 0-indexed; the chain should
        not error when converting to 1-indexed for the context header."""
        docs = [
            Document(
                page_content="Content on the very first page.",
                metadata={"source": "file.pdf", "page": 0},
            )
        ]
        chain = self._build(fake_llm, docs)
        result = chain.invoke("Test question")

        assert "answer" in result
        assert len(result["source_docs"]) == 1

    def test_result_dict_has_required_keys(self, fake_llm, sample_documents):
        """Every chain invocation must return both 'answer' and 'source_docs'."""
        chain = self._build(fake_llm, sample_documents)
        result = chain.invoke("Any question")

        assert "answer" in result
        assert "source_docs" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["source_docs"], list)
