"""Embedding model wrapper with batched ingestion and retry logic.

NVIDIA's API enforces a per-request batch limit (96 texts) and will return
500 errors if you exceed it or hit a rate limit.  This module wraps
NVIDIAEmbeddings to embed in safe batches with exponential-backoff retries.
"""

import time
from typing import List

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document

from config import NVIDIA_API_KEY, NVIDIA_EMBEDDING_MODEL

# Maximum number of texts per API call — stay safely under NVIDIA's limit
EMBED_BATCH_SIZE = 50

# Retry settings
MAX_RETRIES = 4
RETRY_BASE_DELAY = 3   # seconds — doubles on each retry


def get_embedding_model() -> "BatchedNVIDIAEmbeddings":
    """Return a batching-aware embedding model instance."""
    return BatchedNVIDIAEmbeddings(
        model=NVIDIA_EMBEDDING_MODEL,
        api_key=NVIDIA_API_KEY,
    )


class BatchedNVIDIAEmbeddings(NVIDIAEmbeddings):
    """NVIDIAEmbeddings subclass that splits large document lists into
    safe-sized batches and retries on transient 500 errors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* in batches of EMBED_BATCH_SIZE with retries."""
        all_embeddings: List[List[float]] = []
        total   = len(texts)
        batches = [
            texts[i : i + EMBED_BATCH_SIZE]
            for i in range(0, total, EMBED_BATCH_SIZE)
        ]

        print(f"[embeddings] Embedding {total} chunk(s) in "
              f"{len(batches)} batch(es) of <= {EMBED_BATCH_SIZE}.")

        for batch_idx, batch in enumerate(batches, 1):
            print(f"[embeddings] Batch {batch_idx}/{len(batches)} "
                  f"({len(batch)} texts)…", flush=True)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    embeddings = super().embed_documents(batch)
                    all_embeddings.extend(embeddings)
                    break                          # success — move to next batch
                except Exception as exc:
                    if attempt == MAX_RETRIES:
                        raise RuntimeError(
                            f"Embedding failed after {MAX_RETRIES} attempts "
                            f"(batch {batch_idx}/{len(batches)}): {exc}"
                        ) from exc
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(f"[embeddings] Attempt {attempt} failed — "
                          f"retrying in {delay}s.  Error: {exc}")
                    time.sleep(delay)

            # Small pause between batches to respect rate limits
            if batch_idx < len(batches):
                time.sleep(0.5)

        return all_embeddings
