from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from config import NVIDIA_API_KEY, NVIDIA_EMBEDDING_MODEL

def get_embedding_model():
    return NVIDIAEmbeddings(
        model=NVIDIA_EMBEDDING_MODEL,
        api_key=NVIDIA_API_KEY,
    )
