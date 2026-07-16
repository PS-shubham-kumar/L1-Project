# Software Requirements Specification (SRS)

## Project: L1-Project — RAG-Based Document Question Answering System

**Version:** 1.0 
**Date:** 2025  
**Author:** L1-Project Team  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Architecture](#3-system-architecture)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [External Interface Requirements](#6-external-interface-requirements)
7. [Data Requirements](#7-data-requirements)
8. [Configuration & Environment](#8-configuration--environment)
9. [Dependencies](#9-dependencies)
10. [Directory Structure](#10-directory-structure)
11. [Platform Compatibility](#11-platform-compatibility)
12. [Constraints & Assumptions](#12-constraints--assumptions)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for the L1-Project, a Retrieval-Augmented Generation (RAG) system that enables natural language question answering over large document collections. The initial use case is querying the Indian Constitution (404 pages).

### 1.2 Scope
The system allows users to:
- Ingest documents (PDF, DOCX, TXT) into a persistent vector store
- Ask natural language questions and receive context-grounded answers powered by a large language model (LLM)

### 1.3 Definitions

| Term | Definition |
|------|-----------|
| RAG | Retrieval-Augmented Generation — a technique that retrieves relevant document chunks and feeds them as context to an LLM |
| Chunk | A fixed-size segment of text split from a source document |
| Vector Store | A database that stores text embeddings for semantic similarity search |
| Embedding | A numerical vector representation of text used for similarity matching |
| LLM | Large Language Model — the AI model that generates answers |
| NIM | NVIDIA Inference Microservice — NVIDIA's hosted model API |

---

## 2. Overall Description

### 2.1 Product Perspective
L1-Project is a unified Web and CLI application. It runs a Flask-based web server to serve a modern single-page user interface and expose REST APIs, while maintaining CLI interfaces for local scripts. The system uses NVIDIA NIM API endpoints for both embeddings and LLM inference, and ChromaDB as a local persistent vector store. The system follows a two-phase workflow:

```
Phase 1 — Ingest:  Documents (PDF, DOCX, TXT) → Chunks → Embeddings (baai/bge-m3) → ChromaDB
Phase 2 — Query:   Question → Retrieve Top-K Chunks (similarity) → LLM (Llama 3.1 8B) → Answer + Citations
```

### 2.2 User Classes
- **Developer / Researcher** — executes CLI commands or accesses debug APIs to ingest collections and query the system.
- **End User** — interacts with the chat assistant, uploads documents, and views sources via the web interface.

### 2.3 Operating Environment

The system is compatible with the following operating systems:

| OS | Version | Status |
|----|---------|--------|
| macOS | 12 Monterey and above | ✅ Supported |
| Windows | 10 / 11 (64-bit) | ✅ Supported |
| Linux | Ubuntu 20.04+ / Debian-based | ✅ Supported |

- Python 3.10+
- Internet access (required for NVIDIA NIM API calls)
- Virtual environment (venv / conda)

---

## 3. System Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                        index.html                       │
│                   (Single Page Web UI)                  │
└───────────────────────────┬─────────────────────────────┘
                            │ (REST API via HTTP)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│           (Flask Web Server & CLI Entry)                │
│       /api/query, /api/upload, /api/ingest, ingest()    │
└────────┬──────────────────────────┬────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌──────────────────────┐
│ document_loader │      │     rag_chain.py      │
│  (Ingestion)    │      │   (LangChain Chain)   │
└────────┬────────┘      └──────┬───────────────┘
         │                      │
         ▼                          ▼
┌─────────────────┐      ┌──────────────────────┐
│ embedding_model │      │     retriever.py      │
│     bge-m3      │      │   (Top-K Semantic     │
│  (NVIDIA NIM)   │      │      Search)          │
└────────┬────────┘      └──────┬───────────────┘
         │                      │
         ▼                          ▼
┌─────────────────┐      ┌──────────────────────┐
│   vector_db.py  │◄─────│   vector_db.py        │
│   ChromaDB      │      │   ChromaDB (read)     │
│   (persist)     │      └──────────────────────┘
└─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │     llm_client.py    │
                    │  Llama 3.1 8B        │
                    │  (NVIDIA NIM)        │
                    └──────────────────────┘
```

### 3.2 Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| Entry Point & Server | `app.py` | Runs Flask web server, serves static files, implements REST APIs, parses CLI arguments. |
| Web UI Template | `index.html` | Front-end SPA built with TailwindCSS, dynamic chat list, drag-and-drop ingestion, source drawer. |
| Document Loader | `src/ingestion/document_loader.py` | Loads PDF/DOCX/TXT files, splits into text chunks with config-defined overlap. |
| Embedding Model | `src/embeddings/embedding_model.py` | Generates text embeddings using `baai/bge-m3` via NVIDIA NIM with exponential retry. |
| Vector Store | `src/vectorstore/vector_db.py` | Persists and retrieves embeddings using ChromaDB. Deduplicates files on source & page. Falls back to a mock in-memory store if `chromadb` is missing. |
| Retriever | `src/retriever/retriever.py` | Wraps vectorstore for similarity searches retrieving TOP_K chunks. |
| LLM Client | `src/llm/llm_client.py` | Connects to NVIDIA NIM Llama 3.1 8B Instruct API. |
| RAG Chain | `src/chains/rag_chain.py` | Builds the LangChain LCEL pipeline combining prompt context and source citations. |
| Helper | `src/utils/helper.py` | Utility function to format source citation output for CLI/server logging. |
| Config | `config.py` | Centralised configuration (model names, credentials, chunking thresholds). |
| Evaluation Runner | `tests/evaluate_rag.py` | Runs Ragas-based quality evaluation on a golden dataset and saves results to CSV. |

---

## 4. Functional Requirements

### FR-01: Document Ingestion
- The system shall accept documents in `.pdf`, `.docx`, `.doc`, and `.txt` formats.
- The system shall recursively scan the `./data/raw` directory for supported files.
- The system shall split documents into chunks of configurable size (default: 1200 chars) with configurable overlap (default: 200 chars).
- The system shall generate embeddings for each chunk using the `baai/bge-m3` model via NVIDIA NIM APIs.
- The system shall persist all embeddings to a local ChromaDB vector store at `./vectorstore/chroma_db` and deduplicate chunks based on source path and page number.

### FR-02: Question Answering
- The system shall accept natural language questions via web interface queries or CLI arguments.
- The system shall retrieve the top-K (default: 20) most semantically relevant chunks from the vector store.
- The system shall construct a prompt combining retrieved context blocks and the user question.
- The system shall send the prompt to NVIDIA NIM Llama 3.1 8B Instruct and return the generated answer.
- If the answer cannot be determined from context, the system shall respond with: `"I don't have enough information in the indexed documents to answer this."`

### FR-03: CLI Interface
- The system shall support `python app.py ingest` to trigger directory scanning and database ingestion.
- The system shall support `python app.py query "<question>"` to trigger question answering from the terminal.
- The system shall support `python app.py evaluate` to run Ragas pipeline evaluation from the terminal.
- The system shall display usage instructions when incorrect arguments are passed.
- The system shall resolve local directories cross-platform using `os.path.dirname(__file__)`.

### FR-04: Source Formatting
- The system shall format and display source citations (source document name and page number) for retrieved chunks.

### FR-05: Web Interface & REST API
- The system shall serve a responsive Single Page Application UI on `http://127.0.0.1:5000` when `python app.py` is started.
- The web UI shall support drag-and-drop file uploading for all supported file formats.
- The system shall automatically parse and ingest uploaded files into the vector database.
- The web UI shall show a chat dialog interface with typewriter-like typing states, historical query records, and a side-drawer showing retrieved document snippets.
- The backend shall expose API endpoints: `/api/status`, `/api/query`, `/api/ingest`, `/api/upload`, `/api/debug`, `/api/clear`, and `/api/evaluate`.

### FR-06: Database Reset / Clearing
- The system shall allow clearing all indexed documents and wiping the vector database clean via a `/api/clear` POST request.

### FR-07: Quality Evaluation
- The system shall support scoring answer generation using Ragas metric evaluation (faithfulness, answer relevancy, context precision, context recall) over a golden test dataset.

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Ingestion of large document collections (e.g., 400+ pages) shall complete within limits set by the NVIDIA NIM rate caps.
- Embeddings are generated in batch chunks of 50 to prevent HTTP API timeouts.
- Query response times shall generally fall within 2–10 seconds, depending on API network latency.

### 5.2 Scalability
- The chunking and vector store design supports documents of arbitrary length.
- ChromaDB supports local collections of millions of vectors.

### 5.3 Reliability
- The system shall skip unsupported file types without crashing or interrupting ingestion.
- Network and transient API errors shall trigger up to 4 retries with exponential backoff.
- The system shall fall back to an in-memory mock store if `chromadb` is not installed on the system.

### 5.4 Security
- The NVIDIA API key shall be stored in a local `.env` file and never committed to version control.
- Input files are parsed and secured against directory traversal attacks via `secure_filename`.

### 5.5 Maintainability
- All parameters (chunk parameters, DB paths, model names) are centralized in `config.py`.
- Low-coupling modules enable replacing individual ingestion, embedding, or database layers independently.

### 5.6 Portability
- The application runs natively on macOS, Windows, and Linux.
- Path operations utilize `os.path.join` for cross-platform compatibility.

---

## 6. External Interface Requirements

### 6.1 NVIDIA NIM API

| Property | Value |
|----------|-------|
| LLM Model | `meta/llama-3.1-8b-instruct` |
| Embedding Model | `baai/bge-m3` |
| Auth | Bearer token via `NVIDIA_API_KEY` |
| LLM Temperature | 0.7 |

### 6.2 ChromaDB (Local)

| Property | Value |
|----------|-------|
| Type | Local persistent store |
| Path | `./vectorstore/chroma_db` |
| Collection | `rag_collection` |

### 6.3 REST API Endpoints

| Endpoint | Method | Payload / Arguments | Description |
|----------|--------|---------------------|-------------|
| `/` | `GET` | None | Serves the frontend web UI `index.html`. |
| `/api/status` | `GET` | None | Returns database index state, active model, and summary of indexed docs. |
| `/api/query` | `POST` | `{"question": "string"}` | Executes search and returns the AI answer alongside retrieved sources. |
| `/api/ingest` | `POST` | None | Re-indexes all files currently inside the local `data/raw` folder. |
| `/api/upload` | `POST` | `files` (Multipart form-data) | Uploads files to raw storage subdirs and triggers auto-ingestion. |
| `/api/debug` | `GET` | `?q=query_string` (Optional) | Returns chunk diagnostics, list of indexed documents, and optional retrieval test. |
| `/api/clear` | `POST` | None | Deletes all source files from raw folders and wipes ChromaDB vector database. |
| `/api/evaluate` | `POST` | None | Evaluates RAG performance metrics using Ragas framework and returns summary scores. |

### 6.4 Platform-Specific Setup

#### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py            # Start web UI server
python app.py ingest     # Or run ingest via CLI
python app.py evaluate   # Run Ragas evaluation
```

#### Windows (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py            # Start web UI server
python app.py ingest     # Or run ingest via CLI
python app.py evaluate   # Run Ragas evaluation
```

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py            # Start web UI server
python app.py ingest     # Or run ingest via CLI
python app.py evaluate   # Run Ragas evaluation
```

---

## 7. Data Requirements

### 7.1 Input Documents
- Supported formats: `.pdf`, `.docx`, `.doc`, `.txt`
- Location: `./data/raw/` (subdirectories supported: `pdfs/`, `docs/`, `txt/`)
- Current dataset: Indian Constitution PDF (404 pages)

### 7.2 Chunking Parameters

| Parameter | Value |
|-----------|-------|
| Chunk Size | 1200 characters |
| Chunk Overlap | 200 characters |
| Splitter | `RecursiveCharacterTextSplitter` |

### 7.3 Retrieval Parameters

| Parameter | Value |
|-----------|-------|
| Top-K | 20 chunks per query |
| Search Type | Semantic similarity (cosine) |

---

## 8. Configuration & Environment

### 8.1 Environment Variables (`.env`)

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | NVIDIA NIM API key (starts with `nvapi-`) |

### 8.2 `config.py` Parameters

| Constant | Default | Description |
|----------|---------|-------------|
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | LLM model name |
| `NVIDIA_EMBEDDING_MODEL` | `baai/bge-m3` | Embedding model name |
| `CHROMA_DB_PATH` | `./vectorstore/chroma_db` | Vector store path |
| `COLLECTION_NAME` | `rag_collection` | ChromaDB collection name |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `20` | Number of chunks to retrieve |

---

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core LangChain framework |
| `langchain-community` | Community loaders (PDF, DOCX, TXT) |
| `langchain-nvidia-ai-endpoints` | NVIDIA NIM LLM + Embeddings integration |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `chromadb` | Local vector store |
| `langchain-chroma` | LangChain integration for ChromaDB |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `python-dotenv` | `.env` file loading |
| `tiktoken` | Token counting |
| `flask` / `flask-cors` | Web server and API endpoints |
| `pandas` / `numpy` | Data handling utilities |
| `ragas` | Evaluation framework for RAG pipelines |
| `datasets` | Dataset format wrapper required by Ragas |

---

## 10. Directory Structure

```
L1-Project/
├── app.py                        # Flask server & CLI entry point
├── config.py                     # Centralised configuration
├── index.html                    # Frontend single-page app
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed)
├── .gitignore
├── data/
│   └── raw/
│       ├── pdfs/                 # Raw PDF storage (e.g. Indian constitution.pdf)
│       ├── docs/                 # Raw DOCX/DOC storage
│       └── txt/                  # Raw TXT storage
├── src/
│   ├── chains/
│   │   └── rag_chain.py          # LangChain LCEL RAG pipeline
│   ├── embeddings/
│   │   └── embedding_model.py    # NVIDIA NIM Embeddings client with retry logic
│   ├── ingestion/
│   │   └── document_loader.py    # Document loading + recursive splitting
│   ├── llm/
│   │   └── llm_client.py         # NVIDIA NIM LLM client setup
│   ├── retriever/
│   │   └── retriever.py          # Vector store retriever wrapper
│   ├── utils/
│   │   └── helper.py             # CLI formatting helpers
│   └── vectorstore/
│       └── vector_db.py          # ChromaDB read/write interface & mock fallback
├── vectorstore/
│   └── chroma_db/                # Persisted local ChromaDB files (git ignored)
├── notebooks/                    # Jupyter notebooks (exploratory)
└── tests/                        # Test suite
    ├── evaluate_rag.py           # Ragas evaluation pipeline script
    ├── test_embedding.py         # Placeholder test
    ├── test_loader.py            # Placeholder test
    └── test_retrieval.py         # Placeholder test
```

---

## 11. Platform Compatibility

### 11.1 macOS
- Fully tested and supported (macOS 12 Monterey and above)
- Activate virtual environment with: `source venv/bin/activate`
- No additional configuration required

### 11.2 Windows
- Fully compatible — all file paths use `os.path.join()` and `os.path.dirname(__file__)` for cross-platform safety
- Activate virtual environment with: `venv\Scripts\activate` (CMD) or `.\venv\Scripts\Activate.ps1` (PowerShell)
- Ensure Python is added to system `PATH` during installation (check "Add Python to PATH" in the installer)
- If `pip` is not recognised, use `python -m pip install -r requirements.txt`
- Long path support may need to be enabled on Windows 10/11 for deeply nested project directories:
  ```powershell
  # Run in PowerShell as Administrator
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```

### 11.3 Linux
- Compatible with Ubuntu 20.04+ and Debian-based distributions
- Activate virtual environment with: `source venv/bin/activate`
- May require `python3` and `pip3` instead of `python` and `pip` depending on the distro

---

## 12. Constraints & Assumptions

- **Internet Required:** All LLM and embedding calls are made to NVIDIA NIM's hosted API; offline operation is not supported
- **API Key:** A valid `nvapi-` prefixed NVIDIA API key must be present in `.env`
- **Re-ingestion Required:** If `CHUNK_SIZE`, `CHUNK_OVERLAP`, or the embedding model is changed, the vector store must be wiped and re-ingested
- **Single Collection:** All documents share one ChromaDB collection; there is no per-document isolation
- **UI Availability:** The server hosts a local SPA at port 5000, requiring local or networked browser access.
- **Language:** Documents and queries are assumed to be in English.
- **Cross-Platform:** The project is compatible with macOS, Windows, and Linux with no code changes required.
