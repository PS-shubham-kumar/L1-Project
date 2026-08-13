"""Shared test configuration and fixtures for the L1-Project test suite.

Sets up sys.path so that ``src/`` modules are importable and provides
reusable fixtures (FakeChatModel, sample documents) used across test files.
"""

import os
import sys
from typing import Any, List, Optional

import pytest

# ── Make project modules importable (mirrors app.py's sys.path setup) ─────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ── Fake LLM — runs offline, no NVIDIA API key required ──────────────────

class FakeChatModel(BaseChatModel):
    """Deterministic chat model for offline unit tests.

    Returns a fixed string for every prompt so tests can assert on output
    structure without making real API calls.
    """

    fixed_response: str = "Test answer from fake LLM."

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.fixed_response))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def fake_llm():
    """Return a FakeChatModel instance for use in chain tests."""
    return FakeChatModel()


@pytest.fixture
def sample_documents():
    """Return a list of sample Document objects with realistic metadata."""
    return [
        Document(
            page_content="Article 14 guarantees equality before law and equal protection of laws.",
            metadata={"source": "/data/raw/pdfs/constitution.pdf", "page": 5},
        ),
        Document(
            page_content="Article 21 protects the right to life and personal liberty.",
            metadata={"source": "/data/raw/pdfs/constitution.pdf", "page": 10},
        ),
    ]
