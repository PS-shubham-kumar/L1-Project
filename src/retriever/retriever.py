from config import TOP_K

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
