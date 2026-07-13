# L1-Project — RAG-Based Document Q&A

A Retrieval-Augmented Generation (RAG) system for natural language question answering over large document collections, powered by NVIDIA NIM APIs and ChromaDB.

## How It Works

```
Ingest:  Documents → Chunks → Embeddings (NV-EmbedCode-7B) → ChromaDB
Query:   Question → Retrieve Top-K Chunks → Llama 3.1 70B → Answer
```

## Project Structure

```
L1-Project/
├── app.py                  # CLI entry point
├── config.py               # Centralised configuration
├── data/raw/               # Place input documents here (pdf, docx, txt)
├── vectorstore/            # ChromaDB persistent store (auto-generated)
└── src/
    ├── ingestion/          # Document loading & chunking
    ├── embeddings/         # NVIDIA NV-EmbedCode-7B embeddings
    ├── vectorstore/        # ChromaDB read/write
    ├── retriever/          # Top-K semantic retrieval
    ├── llm/                # NVIDIA NIM Llama 3.1 70B client
    ├── chains/             # LangChain LCEL RAG pipeline
    └── utils/              # Source formatting helpers
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

**Ingest documents**

Place your files (`.pdf`, `.docx`, `.txt`) inside `data/raw/`, then run:

```bash
python app.py ingest
```

**Query**

```bash
python app.py query "What are the fundamental rights?"
```

## Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NVIDIA_MODEL` | `meta/llama-3.1-70b-instruct` | LLM for answer generation |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nv-embedcode-7b-v1` | Embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K` | `3` | Retrieved chunks per query |
| `CHROMA_DB_PATH` | `./vectorstore/chroma_db` | Vector store location |

## Requirements

- Python 3.10+
- Internet access (NVIDIA NIM API)
- NVIDIA API key
