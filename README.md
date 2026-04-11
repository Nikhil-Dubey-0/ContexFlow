# ContexFlow 🔍

A production-grade **Retrieval-Augmented Generation (RAG)** system built from scratch in Python — no LangChain, no abstractions, just pure engineering.

Upload PDFs or DOCX files, ask questions, and get accurate answers with source citations.

🔗 **[Live Demo](https://contexflow-3s2zbeultxbxzo9umyd2km.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | FAISS (semantic) + BM25 (keyword) search, merged with Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | Re-scores retrieved chunks for higher relevance accuracy |
| **Query Rewriting** | LLM rewrites user queries for better retrieval (e.g., "semester 6" → "Semester VI") |
| **Streaming Responses** | Token-by-token streaming like ChatGPT |
| **Chat History** | Follow-up questions understand previous context |
| **Multi-Format** | Supports PDF and DOCX document ingestion |
| **Source Attribution** | Every answer cites the exact document and page number |
| **Dual Interface** | Streamlit chat UI + FastAPI REST endpoints |

---

## 📊 Performance Benchmarks

| Metric | FAISS Only | Hybrid (FAISS + BM25) |
|---|---|---|
| **Recall@5** | 0% | 50% (+50% improvement) |
| **Retrieval Latency** | ~10ms | ~14ms |

| Pipeline Stage | Avg Latency |
|---|---|
| Query Rewriting (LLM) | ~470ms |
| Hybrid Retrieval (FAISS + BM25 + RRF) | ~14ms |
| Cross-Encoder Reranking | ~519ms |
| **Total (excl. generation)** | **~1s** |

> Benchmarked on 622 chunks from 6 documents (158 pages) using CPU inference.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│  Query Processor  │  ← LLM rewrites query for better search
└────────┬─────────┘
         ▼
┌──────────────────────────────────┐
│        Hybrid Retriever          │
│  ┌──────────┐  ┌──────────────┐  │
│  │  FAISS    │  │    BM25      │  │
│  │ (semantic)│  │  (keyword)   │  │
│  └────┬─────┘  └──────┬───────┘  │
│       └──────┬─────────┘         │
│         RRF Fusion               │
└────────────┬─────────────────────┘
             ▼
┌──────────────────┐
│    Reranker       │  ← Cross-encoder re-scores for accuracy
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Generator      │  ← LLM generates answer with sources
└────────┬─────────┘
         ▼
    Answer + Sources
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Nikhil-Dubey-0/ContexFlow.git
cd ContexFlow
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:

```env
GROQ_API_KEY=your-groq-api-key
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
DATA_DIR=data
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Add Documents

Place PDF or DOCX files in `data/raw/`

### 4. Run Ingestion

```bash
python scripts/ingest_data.py
```

### 5. Launch

**Streamlit UI:**
```bash
streamlit run frontend/app.py
```

**FastAPI (REST API):**
```bash
uvicorn app.main:app --reload
```
API docs at `http://127.0.0.1:8000/docs`

---

## 📁 Project Structure

```
ContexFlow/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints (/query, /health)
│   ├── core/
│   │   └── config.py              # Pydantic settings from .env
│   ├── db/
│   │   └── vector_store.py        # FAISS wrapper with metadata
│   ├── ingestion/
│   │   ├── loader.py              # PDF + DOCX loading
│   │   ├── chunking.py            # Recursive character splitting
│   │   └── embedding_pipeline.py  # Orchestrates load → chunk → embed → store
│   ├── models/
│   │   ├── embeddings.py          # Sentence-transformer singleton
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── retriever.py           # Hybrid FAISS + BM25 retrieval
│   │   ├── generator.py           # Groq LLM generation (batch + streaming)
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_processor.py     # Query rewriting
│   │   └── rag_pipeline.py        # Main orchestrator
│   ├── utils/
│   │   └── text_cleaning.py       # Regex-based text normalization
│   └── main.py                    # FastAPI app entry point
├── frontend/
│   └── app.py                     # Streamlit chat interface
├── scripts/
│   ├── ingest_data.py             # CLI ingestion script
│   └── test_rag.py                # Quick test script
├── data/
│   ├── raw/                       # Source documents (PDF, DOCX)
│   └── embeddings/                # FAISS index + metadata
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Llama 3.3 70B via Groq (free tier) |
| **Embeddings** | `all-MiniLM-L6-v2` (local, CPU) |
| **Vector DB** | FAISS (local) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **PDF Processing** | PyMuPDF (fitz) |
| **DOCX Processing** | python-docx |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Config** | Pydantic Settings + dotenv |

---

## 🧠 Why No LangChain?

This project intentionally avoids LangChain and similar frameworks to:

1. **Demonstrate engineering depth** — every component is written and understood
2. **Minimize dependency bloat** — only use what you need
3. **Full control** — easy to debug, modify, and extend
4. **Interview-ready** — can explain every line of the pipeline

---

## 📝 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a question, get answer + sources |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Edge AI?", "top_k": 5}'
```

---

## 📄 License

MIT
