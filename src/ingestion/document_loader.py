from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os

def load_documents(data_dir: str = "./data/raw") -> list:
    docs = []
    loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in loaders:
                path = os.path.join(root, file)
                docs.extend(loaders[ext](path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)
