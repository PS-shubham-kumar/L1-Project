from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PROMPT_TEMPLATE = """Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def build_rag_chain(llm, retriever):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

