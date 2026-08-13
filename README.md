# L1-Project — RAG-Based Document Q&A

## Problem Statement & Use Case

### Problem Statement
Navigating, searching, and extracting accurate information from large, complex legal document collections—such as the 404-page *Constitution of India*—is time-consuming and error-prone using standard keyword search. Users face several key challenges:
- **Context Loss**: Traditional keyword queries miss semantically relevant passages when legal concepts are phrased differently.
- **Hallucination Risk**: Standard LLMs can fabricate legal clauses or cite incorrect articles when generating responses without strict context grounding.
- **Lack of Traceability**: Users require exact source document and page-level citations (e.g., `[Source: constitution.pdf, page 5]`) to verify claims against authoritative legal texts.

### Solution
This repository delivers an end-to-end, context-grounded **Retrieval-Augmented Generation (RAG)** assistant powered by **NVIDIA NIM APIs** (`nvidia/nv-embed-v1` dense embeddings & `meta/llama-3.1-8b-instruct`) and **ChromaDB** vector database. The system ingests multi-format documents (PDF, DOCX, TXT), splits them into semantic chunks, enforces similarity thresholding to filter out low-relevance noise, and synthesizes accurate, factual answers bound strictly to retrieved context with full citation tracking.


## How It Works

```
Ingest:  Documents → Chunks → Embeddings (nv-embed-v1) → ChromaDB
Query:   Question → Retrieve Top-K Chunks (similarity >= 0.3) → Llama 3.1 8B → Answer + Citations
```

## Project Structure

```
L1-Project/
├── app.py                  # Web server & CLI entry point
├── config.py               # Centralised configuration & rationale
├── index.html              # Frontend single-page app (UI)
├── SRS.md                  # Software Requirements Specification
├── README.md               # Project documentation
├── .env.example            # Environment variable configuration template
├── requirements.txt        # Python dependencies
├── data/raw/               # Input documents store (pdf, docx, txt)
├── vectorstore/            # ChromaDB persistent store (auto-generated)
├── src/
│   ├── ingestion/          # Document loading & chunking
│   ├── embeddings/         # NVIDIA nv-embed-v1 embeddings
│   ├── vectorstore/        # ChromaDB read/write (vector_db.py)
│   ├── retriever/          # Top-K semantic retrieval with threshold gate
│   ├── llm/                # NVIDIA NIM Llama 3.1 8B client
│   ├── chains/             # LangChain LCEL RAG pipeline
│   └── utils/              # Source formatting helpers
└── tests/                  # Evaluation pipeline & test suite
    ├── conftest.py         # Pytest fixtures & fake LLM wrapper
    ├── test_document_loader.py # Loader & chunking unit tests
    ├── test_rag_chain.py   # RAG chain unit tests
    ├── test_retriever.py   # Retriever configuration unit tests
    ├── test_vector_db.py   # Vector DB store unit tests
    ├── evaluate_rag.py     # Ragas evaluation script
    └── evaluation_report.csv # Ragas evaluation output report
```

## Setup

**1. Clone & create virtual environment**

```bash
# macOS / Linux
python -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

Copy `.env.example` to `.env` and configure your API key:

```bash
# macOS / Linux / Windows PowerShell
cp .env.example .env
```

Edit `.env`:

```ini
NVIDIA_API_KEY=<your_nvidia_api_key>
```

Get your API key at [build.nvidia.com](https://build.nvidia.com).

## Usage

**Start Web Server (UI)**

```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

**Ingest documents via CLI**

Place your files (`.pdf`, `.docx`, `.txt`) inside `data/raw/`, then run:

```bash
python app.py ingest
```

### Sample Dataset & Source Document

The system is configured with official document data for question answering over Indian law:
- **Document Title:** Constitution of India
- **Local Storage Path:** [`data/raw/pdfs/the_constitution_of_india.pdf`](file:///c:/Users/ShubhamKumar/Desktop/L1-Project/data/raw/pdfs/the_constitution_of_india.pdf)
- **Official Portal Link:** [Legislative Department, Ministry of Law and Justice](https://legislative.gov.in/constitution-of-india/)
- **Direct PDF Download:** [Constitution of India PDF (Official Link)](https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf)


**Query via CLI**

```bash
python app.py query "What are the fundamental rights?"
```

**Run Unit Test Suite**

```bash
pytest
```

**Run RAG Evaluation Pipeline**

```bash
python app.py evaluate
```

## Configuration & Hyperparameter Rationale

All system parameters are defined in `config.py`:

| Parameter | Default Value | Technical Description | Design & Trade-off Rationale |
|-----------|---------------|-----------------------|------------------------------|
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | LLM for answer generation | 8B parameter instruction-tuned model provides low latency while strictly adhering to context grounding system prompts. |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embed-v1` | Dense text embeddings | 4096-dimensional dense embedding model providing high semantic resolution for legal and technical document retrieval. |
| `CHUNK_SIZE` | `500` | Characters per text chunk | Optimized to capture single legal articles or complete paragraphs (e.g. Indian Constitution articles) without diluting focus or exceeding context windows. |
| `CHUNK_OVERLAP` | `50` | Overlap characters | 10% overlap preserves semantic continuity across sentence/chunk split boundaries. |
| `TOP_K` | `20` | Max candidate chunks retrieved | Ensures broad candidate recall across multi-page document collections before relevance filtering. |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity gate | Enforces similarity thresholding via ChromaDB cosine distance (`hnsw:space: cosine`) to filter out weak noise before passing context to LLM. |
| `CHROMA_DB_PATH` | `./vectorstore/chroma_db` | Vector store storage directory | Absolute path configuration allowing reliable SQLite index persistence regardless of CWD. |

## Requirements

- Python 3.10+
- Internet access (NVIDIA NIM API)
- NVIDIA API key
