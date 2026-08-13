from config import TOP_K, SIMILARITY_THRESHOLD


def get_retriever(vectorstore):
    """Return a retriever that fetches up to TOP_K chunks above SIMILARITY_THRESHOLD.

    Uses ``similarity_score_threshold`` search so that low-relevance chunks are
    dropped before they reach the LLM context window.  This prevents the model
    from being flooded with weak matches when TOP_K is large.
    """
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": SIMILARITY_THRESHOLD,
            "k": TOP_K,
        },
    )

