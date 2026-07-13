from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Be concise and factual."""

FIXED_CONTEXT = "This document is about Indian Constitution."

def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    store = {"docs": []}

    def format_docs(docs):
        store["docs"] = docs
        retrieved = "\n\n".join(doc.page_content for doc in docs)
        return f"{FIXED_CONTEXT}\n\n{retrieved}".strip() if FIXED_CONTEXT else retrieved

    answer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def invoke_with_sources(question: str) -> dict:
        answer = answer_chain.invoke(question)
        return {"answer": answer, "source_docs": store["docs"]}

    return RunnableLambda(invoke_with_sources)
