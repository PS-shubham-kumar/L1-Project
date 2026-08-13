"""Tests for retriever configuration (search type, threshold, top-k).

Uses a mocked vectorstore to verify that ``get_retriever`` applies the
correct search strategy and parameters from config.py.
"""

from unittest.mock import MagicMock

from retriever.retriever import get_retriever
from config import TOP_K, SIMILARITY_THRESHOLD


class TestRetrieverConfig:
    """Verify the retriever is configured with the expected search parameters."""

    def test_uses_similarity_score_threshold_search(self):
        mock_vs = MagicMock()
        get_retriever(mock_vs)

        call_kwargs = mock_vs.as_retriever.call_args[1]
        assert call_kwargs["search_type"] == "similarity_score_threshold"

    def test_passes_score_threshold_from_config(self):
        mock_vs = MagicMock()
        get_retriever(mock_vs)

        search_kwargs = mock_vs.as_retriever.call_args[1]["search_kwargs"]
        assert search_kwargs["score_threshold"] == SIMILARITY_THRESHOLD

    def test_passes_top_k_from_config(self):
        mock_vs = MagicMock()
        get_retriever(mock_vs)

        search_kwargs = mock_vs.as_retriever.call_args[1]["search_kwargs"]
        assert search_kwargs["k"] == TOP_K

    def test_returns_retriever_from_vectorstore(self):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        result = get_retriever(mock_vs)
        assert result is mock_retriever
