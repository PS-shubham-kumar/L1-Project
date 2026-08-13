# L1-Project — RAG-Based Document Q&A

A Retrieval-Augmented Generation (RAG) system for natural language question answering over large document collections, powered by NVIDIA NIM APIs and ChromaDB.

## How It Works

```
Ingest:  Documents → Chunks → Embeddings (nv-embed-v1) → ChromaDB
Query:   Question → Retrieve Top-K Chunks → Llama 3.1 8B → Answer
```

## Project Structure

```
L1-Project/
├── app.py                  # Web server & CLI entry point
├── config.py               # Centralised configuration
├── index.html              # Frontend single-page app (UI)
├── SRS.md                  # Software Requirements Specification
├── requirements.txt        # Python dependencies
├── data/raw/               # Input documents store (pdf, docx, txt)
├── vectorstore/            # ChromaDB persistent store (auto-generated)
├── src/
│   ├── ingestion/          # Document loading & chunking
│   ├── embeddings/         # NVIDIA nv-embed-v1 embeddings
│   ├── vectorstore/        # ChromaDB read/write
│   ├── retriever/          # Top-K semantic retrieval
│   ├── llm/                # NVIDIA NIM Llama 3.1 8B client
│   ├── chains/             # LangChain LCEL RAG pipeline
│   └── utils/              # Source formatting helpers
└── tests/                  # Evaluation pipeline & benchmarks
    ├── evaluate_rag.py     # Ragas evaluation script
    └── evaluation_report.csv # Evaluation report metrics
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

Create a `.env` file in the project root:

```
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

**Query via CLI**

```bash
python app.py query "What are the fundamental rights?"
```

**Run RAG Evaluation**

```bash
python app.py evaluate
```

## Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | LLM for answer generation |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embed-v1` | Embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `20` | Retrieved chunks per query |
| `CHROMA_DB_PATH` | `./vectorstore/chroma_db` | Vector store location |

## Requirements

- Python 3.10+
- Internet access (NVIDIA NIM API)
- NVIDIA API key
