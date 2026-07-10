def format_sources(source_docs: list) -> str:
    seen = set()
    sources = []
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return "\n".join(f"  - {s}" for s in sources)
