import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Model Specifications
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"  # 8B instruction-tuned LLM optimized for grounded Q&A
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embed-v1"  # 4096-dim high-capacity dense embedding model

# Use an absolute path so ChromaDB can write its SQLite file regardless of CWD
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(_PROJECT_ROOT, "vectorstore", "chroma_db")
COLLECTION_NAME = "rag_collection"

# Ingestion & Chunking Rationale:
# CHUNK_SIZE = 500 chars captures standard paragraphs/articles (e.g. Indian Constitution articles)
# without overflowing context limits or splitting core semantic concepts.
# CHUNK_OVERLAP = 50 chars ensures boundary sentences preserve context across chunk splits.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Rationale:
# TOP_K = 20 allows retrieving sufficient candidate context segments across multi-page documents.
# SIMILARITY_THRESHOLD = 0.3 enforces a cosine distance relevance gate (0-1 similarity scale)
# to drop noisy/unrelated chunks before passing context to the LLM.
TOP_K = 20
SIMILARITY_THRESHOLD = 0.3

