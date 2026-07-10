import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedcode-7b-v1"

CHROMA_DB_PATH = "./vectorstore/chroma_db"
COLLECTION_NAME = "rag_collection"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
TOP_K = 8
