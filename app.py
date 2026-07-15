"""
L1-Project RAG Application
==========================
Single entry point for both the web UI and CLI.

Web server (default):
    python app.py

Ingest documents from data/raw:
    python app.py ingest

Run a one-off query from the terminal:
    python app.py query "What is Article 14?"
"""

import os
import sys
import traceback
import certifi
from werkzeug.utils import secure_filename

# Fix SSL certificate verification for NVIDIA API calls
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Make src/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ingestion.document_loader import load_documents
from embeddings.embedding_model import get_embedding_model
from vectorstore.vector_db import get_vectorstore
from retriever.retriever import get_retriever
from llm.llm_client import get_llm
from chains.rag_chain import build_rag_chain
from utils.helper import format_sources
from config import CHROMA_DB_PATH, COLLECTION_NAME, NVIDIA_MODEL

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
CORS(app)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    "pdf":  "pdfs",
    "docx": "docs",
    "doc":  "docs",
    "txt":  "txt",
}

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _save_dir(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    subdir = ALLOWED_EXTENSIONS.get(ext, "pdfs")
    path = os.path.join(os.path.dirname(__file__), "data", "raw", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _build_chain():
    """Initialise embedding model, vector store, retriever and LLM chain."""
    embeddings  = get_embedding_model()
    vectorstore = get_vectorstore(embeddings)
    retriever   = get_retriever(vectorstore)
    llm         = get_llm()
    return build_rag_chain(llm, retriever)


def _detect_document_name() -> str:
    """Return a summary of all indexed source files across data/raw subdirs."""
    raw_base = os.path.join(os.path.dirname(__file__), "data", "raw")
    all_files = []
    for subdir in ("pdfs", "docs", "txt"):
        folder = os.path.join(raw_base, subdir)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.startswith("."):
                all_files.append(os.path.splitext(f)[0].replace("_", " ").title())
    if not all_files:
        return "No documents indexed"
    if len(all_files) == 1:
        return all_files[0]
    return f"{all_files[0]} +{len(all_files) - 1} more"


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def api_status():
    """Return index status and model info."""
    try:
        db_path = CHROMA_DB_PATH
        indexed = os.path.isdir(db_path)
        return jsonify({
            "ok":         True,
            "indexed":    indexed,
            "collection": COLLECTION_NAME,
            "model":      NVIDIA_MODEL,
            "document":   _detect_document_name(),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/query", methods=["POST"])
def api_query():
    """Run a RAG query and return the answer plus source metadata."""
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    try:
        chain  = _build_chain()
        result = chain.invoke(question)

        sources = []
        seen    = set()
        for doc in result.get("source_docs", []):
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            key  = (src, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source":  os.path.basename(src),
                    "page":    (page + 1) if page is not None else None,
                    "snippet": doc.page_content[:300],
                })

        return jsonify({"ok": True, "answer": result["answer"], "sources": sources})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Ingest all documents found under data/raw into ChromaDB."""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
        docs     = load_documents(data_dir)
        if not docs:
            return jsonify({"ok": False, "error": "No documents found in data/raw"}), 400

        embeddings = get_embedding_model()
        get_vectorstore(embeddings, docs)

        return jsonify({
            "ok":      True,
            "chunks":  len(docs),
            "message": f"Successfully indexed {len(docs)} chunks.",
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept one or more uploaded files, save them to data/raw/<subdir>,
    then immediately ingest the whole data/raw tree into ChromaDB.

    Accepted types: .pdf  → data/raw/pdfs/
                    .docx / .doc → data/raw/docs/
                    .txt  → data/raw/txt/
    """
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"ok": False, "error": "No files received."}), 400

    saved = []
    rejected = []
    for f in files:
        fname = secure_filename(f.filename)
        if not fname or not _allowed(fname):
            rejected.append(f.filename)
            continue
        dest = os.path.join(_save_dir(fname), fname)
        f.save(dest)
        saved.append(fname)

    if not saved:
        return jsonify({
            "ok":    False,
            "error": f"No supported files uploaded. Rejected: {', '.join(rejected)}. "
                     "Accepted formats: PDF, DOCX, DOC, TXT.",
        }), 400

    # Ingest the full data/raw tree so everything stays in sync
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
        docs     = load_documents(data_dir)
        embeddings = get_embedding_model()
        get_vectorstore(embeddings, docs)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "ok":    False,
            "saved": saved,
            "error": f"Files saved but indexing failed: {exc}",
        }), 500

    return jsonify({
        "ok":      True,
        "saved":   saved,
        "chunks":  len(docs),
        "message": f"Uploaded {len(saved)} file(s) and indexed {len(docs)} chunks.",
        **({"rejected": rejected} if rejected else {}),
    })


@app.route("/api/debug", methods=["GET"])
def api_debug():
    """Return a snapshot of what is actually stored in ChromaDB.
    Useful for diagnosing retrieval issues.
    GET /api/debug?q=<optional query>
    """
    try:
        embeddings  = get_embedding_model()
        vectorstore = get_vectorstore(embeddings)

        # Total chunks
        raw   = vectorstore.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        total = len(metas)

        # Unique source files
        sources = sorted({
            os.path.basename(m.get("source", "unknown"))
            for m in metas
        })

        result = {
            "ok":           True,
            "total_chunks": total,
            "sources":      sources,
        }

        # Optionally run a test retrieval
        q = request.args.get("q", "").strip()
        if q:
            retriever = get_retriever(vectorstore)
            docs      = retriever.invoke(q)
            result["test_query"]  = q
            result["retrieved_k"] = len(docs)
            result["retrieved_sources"] = [
                {
                    "source": os.path.basename(d.metadata.get("source", "unknown")),
                    "page":   d.metadata.get("page"),
                    "chars":  len(d.page_content),
                }
                for d in docs
            ]

        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500



def api_clear():
    """
    Delete all source files from data/raw/ and wipe the ChromaDB vector store.
    This resets the knowledge base completely.
    """
    import shutil

    errors = []

    # 1. Remove all files from every raw subdirectory (keep the folders)
    raw_base = os.path.join(os.path.dirname(__file__), "data", "raw")
    for subdir in ("pdfs", "docs", "txt"):
        folder = os.path.join(raw_base, subdir)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.startswith("."):
                continue  # keep .gitkeep etc.
            fpath = os.path.join(folder, fname)
            try:
                os.remove(fpath)
            except Exception as exc:
                errors.append(f"{fname}: {exc}")

    # 2. Wipe the ChromaDB persisted store
    db_path = CHROMA_DB_PATH
    if os.path.isdir(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception as exc:
            errors.append(f"ChromaDB: {exc}")

    if errors:
        return jsonify({"ok": False, "error": "Partial clear. " + "; ".join(errors)}), 500

    return jsonify({"ok": True, "message": "Knowledge base cleared successfully."})




def cli_ingest(data_dir: str = "./data/raw") -> None:
    print("[ingest] Loading and splitting documents…")
    docs = load_documents(data_dir)
    print(f"[ingest] {len(docs)} chunks created.")
    print("[ingest] Building vector store…")
    embeddings = get_embedding_model()
    get_vectorstore(embeddings, docs)
    print("[ingest] Done — vector store saved.")


def cli_query(question: str) -> None:
    chain  = _build_chain()
    result = chain.invoke(question)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:\n{format_sources(result['source_docs'])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Default: start the web server
        print("[server] Starting L1-Project RAG on http://127.0.0.1:5000")
        app.run(debug=False, host="0.0.0.0", port=5000)

    elif args[0] == "ingest":
        cli_ingest()

    elif args[0] == "query" and len(args) == 2:
        cli_query(args[1])

    else:
        print("Usage:")
        print("  python app.py                     — start web server")
        print("  python app.py ingest              — ingest data/raw documents")
        print('  python app.py query "<question>"  — run a one-off query')
        sys.exit(1)
