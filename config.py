import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embed-v1"  # 4096-dim, verified working

CHROMA_DB_PATH = "./vectorstore/chroma_db"
COLLECTION_NAME = "rag_collection"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 20  # Retrieve up to 20 chunks across all indexed documents
