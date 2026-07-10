from langchain_nvidia_ai_endpoints import ChatNVIDIA
from config import NVIDIA_API_KEY, NVIDIA_MODEL

def get_llm():
    return ChatNVIDIA(
        model=NVIDIA_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0.7,
    )
