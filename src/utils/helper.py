def format_sources(source_docs: list) -> str:
    seen = set()
    sources = []
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            label = f"{src} (page {page + 1})" if page is not None else src
            sources.append(label)
    return "\n".join(f"  - {s}" for s in sources)
