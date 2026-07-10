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
L1-Project is a standalone CLI application. It uses NVIDIA's NIM API for both embeddings and LLM inference, and ChromaDB as a local persistent vector store. The system follows a two-phase workflow:

```
Phase 1 — Ingest:  Documents → Chunks → Embeddings → ChromaDB
Phase 2 — Query:   Question → Retrieve Chunks → LLM → Answer
```

### 2.2 User Classes
- **Developer / Researcher** — runs CLI commands to ingest documents and query them

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
│                        app.py                           │
│              (CLI Entry Point)                          │
│         ingest()              query()                   │
└────────┬──────────────────────────┬────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌──────────────────────┐
│ document_loader │      │     rag_chain.py      │
│  (Ingestion)    │      │   (LangChain Chain)   │
└────────┬────────┘      └──────┬───────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────────────┐
│ embedding_model │      │     retriever.py      │
│ NV-EmbedCode-7B │      │   (Top-K Semantic     │
│  (NVIDIA NIM)   │      │      Search)          │
└────────┬────────┘      └──────┬───────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────────────┐
│   vector_db.py  │◄─────│   vector_db.py        │
│   ChromaDB      │      │   ChromaDB (read)     │
│   (persist)     │      └──────────────────────┘
└─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │     llm_client.py    │
                    │  Llama 3.1 70B       │
                    │  (NVIDIA NIM)        │
                    └──────────────────────┘
```

### 3.2 Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| Entry Point | `app.py` | CLI parsing, orchestrates ingest and query flows |
| Document Loader | `src/ingestion/document_loader.py` | Loads PDF/DOCX/TXT files, splits into chunks |
| Embedding Model | `src/embeddings/embedding_model.py` | Generates text embeddings via NVIDIA NV-EmbedCode |
| Vector Store | `src/vectorstore/vector_db.py` | Persists and retrieves embeddings using ChromaDB |
| Retriever | `src/retriever/retriever.py` | Wraps vectorstore for top-K semantic retrieval |
| LLM Client | `src/llm/llm_client.py` | Connects to NVIDIA NIM Llama 3.1 70B |
| RAG Chain | `src/chains/rag_chain.py` | Builds the LangChain LCEL pipeline |
| Helper | `src/utils/helper.py` | Formats source document metadata |
| Config | `config.py` | Centralised configuration and env var loading |

---

## 4. Functional Requirements

### FR-01: Document Ingestion
- The system shall accept documents in `.pdf`, `.docx`, and `.txt` formats
- The system shall recursively scan the `./data/raw` directory for supported files
- The system shall split documents into chunks of configurable size (default: 1000 chars) with configurable overlap (default: 100 chars)
- The system shall generate embeddings for each chunk using NVIDIA NV-EmbedCode-7B
- The system shall persist all embeddings to a local ChromaDB vector store at `./vectorstore/chroma_db`

### FR-02: Question Answering
- The system shall accept a natural language question as a CLI argument
- The system shall retrieve the top-K (default: 5) most semantically relevant chunks from the vector store
- The system shall construct a prompt combining retrieved context and the user question
- The system shall send the prompt to NVIDIA NIM Llama 3.1 70B and return the generated answer
- If the answer cannot be determined from context, the system shall respond with "I don't know"

### FR-03: CLI Interface
- The system shall support `python app.py ingest` to trigger document ingestion
- The system shall support `python app.py query "<question>"` to trigger question answering
- The system shall print usage instructions if incorrect arguments are provided
- The system shall resolve the `src/` path using `os.path.dirname(__file__)` to ensure cross-platform compatibility on both macOS and Windows

### FR-04: Source Formatting
- The system shall be capable of formatting and displaying source document metadata (file paths) for retrieved chunks via `format_sources()`

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Ingestion of a 400-page PDF should complete within a reasonable time bounded by NVIDIA NIM API rate limits
- Query response time is dependent on NVIDIA NIM API latency (typically 2–10 seconds)

### 5.2 Scalability
- The chunking and vector store design supports documents of arbitrary length
- ChromaDB supports collections of millions of vectors locally

### 5.3 Reliability
- The system shall not crash on unsupported file types — unsupported files are silently skipped
- The system shall propagate API errors clearly to the user

### 5.4 Security
- The NVIDIA API key shall be stored in a `.env` file and never hardcoded in source files
- `.env` shall be listed in `.gitignore` to prevent accidental exposure

### 5.5 Maintainability
- All configuration values (model names, paths, chunk sizes) are centralised in `config.py`
- Each module has a single responsibility, making it independently replaceable

### 5.6 Portability
- The system shall run on macOS, Windows, and Linux without any code changes
- All file paths are constructed using `os.path.join()` and `os.path.dirname(__file__)` for OS-agnostic path handling
- No platform-specific shell commands or binaries are used

---

## 6. External Interface Requirements

### 6.1 NVIDIA NIM API

| Property | Value |
|----------|-------|
| LLM Model | `meta/llama-3.1-70b-instruct` |
| Embedding Model | `nvidia/nv-embedcode-7b-v1` |
| Auth | Bearer token via `NVIDIA_API_KEY` |
| LLM Temperature | 0.7 |

### 6.2 ChromaDB (Local)

| Property | Value |
|----------|-------|
| Type | Local persistent store |
| Path | `./vectorstore/chroma_db` |
| Collection | `rag_collection` |

### 6.3 CLI

| Command | Description |
|---------|-------------|
| `python app.py ingest` | Ingest all documents from `./data/raw` |
| `python app.py query "<question>"` | Query the RAG system |

### 6.4 Platform-Specific Setup

#### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py ingest
python app.py query "Your question here"
```

#### Windows (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py ingest
python app.py query "Your question here"
```

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py ingest
python app.py query "Your question here"
```

> **Note for Windows PowerShell users:** If script execution is blocked, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 7. Data Requirements

### 7.1 Input Documents
- Supported formats: `.pdf`, `.docx`, `.txt`
- Location: `./data/raw/` (subdirectories supported)
- Current dataset: Indian Constitution PDF (404 pages)

### 7.2 Chunking Parameters

| Parameter | Value |
|-----------|-------|
| Chunk Size | 1000 characters |
| Chunk Overlap | 100 characters |
| Splitter | `RecursiveCharacterTextSplitter` |

### 7.3 Retrieval Parameters

| Parameter | Value |
|-----------|-------|
| Top-K | 5 chunks per query |
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
| `NVIDIA_MODEL` | `meta/llama-3.1-70b-instruct` | LLM model name |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embedcode-7b-v1` | Embedding model name |
| `CHROMA_DB_PATH` | `./vectorstore/chroma_db` | Vector store path |
| `COLLECTION_NAME` | `rag_collection` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |

---

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core LangChain framework |
| `langchain-community` | Community loaders (PDF, DOCX, TXT) |
| `langchain-nvidia-ai-endpoints` | NVIDIA NIM LLM + Embeddings integration |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `chromadb` | Local vector store |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `python-dotenv` | `.env` file loading |
| `tiktoken` | Token counting |
| `pandas` / `numpy` | Data handling utilities |

---

## 10. Directory Structure

```
L1-Project/
├── app.py                        # CLI entry point
├── config.py                     # Centralised configuration
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed)
├── .gitignore
├── data/
│   └── raw/
│       └── pdfs/
│           └── Indian constitution.pdf
├── src/
│   ├── chains/
│   │   └── rag_chain.py          # LangChain LCEL RAG pipeline
│   ├── embeddings/
│   │   └── embedding_model.py    # NVIDIA NV-EmbedCode embeddings
│   ├── ingestion/
│   │   └── document_loader.py    # Document loading + chunking
│   ├── llm/
│   │   └── llm_client.py         # NVIDIA NIM Llama 3.1 70B client
│   ├── retriever/
│   │   └── retriever.py          # Vector store retriever wrapper
│   ├── utils/
│   │   └── helper.py             # Source formatting utility
│   └── vectorstore/
│       └── vector_db.py          # ChromaDB read/write interface
├── vectorstore/
│   └── chroma_db/                # Persisted vector embeddings
├── notebooks/                    # Jupyter notebooks (exploratory)
└── tests/                        # Test suite
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
- **CLI Only:** The current interface is command-line only; no web UI or REST API is provided
- **Language:** Documents and queries are assumed to be in English
- **Cross-Platform:** The project is compatible with macOS, Windows, and Linux with no code changes required
