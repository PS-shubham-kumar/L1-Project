"""Tests for document loading, chunking, and metadata propagation.

These tests use real temporary files (no mocks) to verify that the ingestion
pipeline correctly loads .txt files, skips unsupported formats, handles
encoding fallbacks, and produces properly-chunked documents with metadata.
"""

import pytest

from ingestion.document_loader import load_documents, _load_file


class TestLoadFile:
    """Unit tests for the _load_file helper."""

    def test_loads_txt_file_successfully(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello world. This is a test document for loading.")
        docs = _load_file(str(f))
        assert len(docs) >= 1
        assert "Hello world" in docs[0].page_content

    def test_unsupported_extension_returns_empty_list(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        assert _load_file(str(f)) == []

    def test_txt_latin1_encoding_fallback(self, tmp_path):
        """If UTF-8 decoding fails, the loader should fall back to latin-1."""
        f = tmp_path / "latin.txt"
        f.write_bytes("café résumé naïve".encode("latin-1"))
        docs = _load_file(str(f))
        assert len(docs) >= 1


class TestLoadDocuments:
    """Integration tests for the full load → chunk pipeline."""

    def test_chunks_carry_source_metadata(self, tmp_path):
        (tmp_path / "info.txt").write_text("Some content for chunking purposes. " * 20)
        chunks = load_documents(str(tmp_path))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "info.txt" in chunk.metadata["source"]

    def test_chunk_size_is_bounded_by_config(self, tmp_path):
        from config import CHUNK_SIZE

        (tmp_path / "big.txt").write_text("word " * 1000)  # ~5000 chars
        chunks = load_documents(str(tmp_path))
        for chunk in chunks:
            # Allow a small margin for the recursive splitter
            assert len(chunk.page_content) <= CHUNK_SIZE + 100

    def test_empty_directory_produces_no_chunks(self, tmp_path):
        assert load_documents(str(tmp_path)) == []

    def test_skips_unsupported_file_types(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Markdown is not supported")
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3")
        assert load_documents(str(tmp_path)) == []

    def test_deterministic_file_ordering(self, tmp_path):
        """Files should be loaded in sorted order for reproducible chunk sequences."""
        (tmp_path / "b_file.txt").write_text("Content B. " * 30)
        (tmp_path / "a_file.txt").write_text("Content A. " * 30)
        chunks = load_documents(str(tmp_path))
        sources = [c.metadata["source"] for c in chunks]
        first_a = next(i for i, s in enumerate(sources) if "a_file" in s)
        first_b = next(i for i, s in enumerate(sources) if "b_file" in s)
        assert first_a < first_b

    def test_multiple_files_are_all_loaded(self, tmp_path):
        (tmp_path / "one.txt").write_text("First document content. " * 10)
        (tmp_path / "two.txt").write_text("Second document content. " * 10)
        chunks = load_documents(str(tmp_path))
        sources = {c.metadata["source"] for c in chunks}
        assert any("one.txt" in s for s in sources)
        assert any("two.txt" in s for s in sources)
