import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestion.document_loader import load_documents
from embeddings.embedding_model import get_embedding_model
from vectorstore.vector_db import get_vectorstore
from retriever.retriever import get_retriever
from llm.llm_client import get_llm
from chains.rag_chain import build_rag_chain
from utils.helper import format_sources

def ingest(data_dir: str = "./data/raw"):
    print("Loading and splitting documents...")
    docs = load_documents(data_dir)
    print(f"  {len(docs)} chunks created.")

    print("Building vector store...")
    embeddings = get_embedding_model()
    get_vectorstore(embeddings, docs)
    print("  Vector store saved.")
    
def query(question: str):
    embeddings = get_embedding_model()
    vectorstore = get_vectorstore(embeddings)
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    chain = build_rag_chain(llm, retriever)

    result = chain.invoke(question)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:\n{format_sources(result['source_docs'])}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py ingest | python app.py query '<question>'")
        sys.exit(1)

    if sys.argv[1] == "ingest":
        ingest()
    elif sys.argv[1] == "query" and len(sys.argv) == 3:
        query(sys.argv[2])
    else:
        print("Usage: python app.py ingest | python app.py query '<question>'")
