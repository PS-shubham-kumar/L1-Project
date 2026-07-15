"""RAG chain — retrieves relevant chunks then calls the LLM."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly \
based on the provided context extracted from the indexed documents.

Rules:
- Answer ONLY from the context below. Do not use any outside knowledge.
- Each context block is prefixed with [Source: <filename>, page <n>].  \
  Cite the source when it adds value to the answer.
- If the answer cannot be found in the context, reply exactly:
  "I don't have enough information in the indexed documents to answer this."
- If the question spans multiple documents, synthesise the answer from all \
  relevant context blocks.
- Be concise, accurate, and factual."""


def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context from indexed documents:\n{context}\n\nQuestion: {question}"),
    ])

    answer_chain = prompt | llm | StrOutputParser()

    def invoke_with_sources(question: str) -> dict:
        # ── Retrieve top-K chunks across ALL indexed documents ────────────
        docs = retriever.invoke(question)

        print(f"[chain] Query: {question!r}")
        print(f"[chain] Retrieved {len(docs)} chunk(s) from the vector store.")

        if not docs:
            return {
                "answer": (
                    "No relevant chunks were retrieved from the vector store. "
                    "Please make sure documents have been ingested first."
                ),
                "source_docs": [],
            }

        # ── Build context with source headers ─────────────────────────────
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            # Normalise path separators (Windows backslash / POSIX slash)
            source_name = source.replace("\\", "/").split("/")[-1]
            page        = doc.metadata.get("page")
            header      = (
                f"[Source: {source_name}, page {page + 1}]"
                if page is not None
                else f"[Source: {source_name}]"
            )
            context_parts.append(f"{header}\n{doc.page_content}")
            print(f"  • {header}  ({len(doc.page_content)} chars)")

        context = "\n\n---\n\n".join(context_parts)

        # ── Call the LLM ──────────────────────────────────────────────────
        answer = answer_chain.invoke({"context": context, "question": question})
        return {"answer": answer, "source_docs": docs}

    return RunnableLambda(invoke_with_sources)
