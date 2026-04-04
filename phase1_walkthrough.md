# ContexFlow — Phase 1: Foundation (Step-by-Step)

**Goal:** Project structure + ingestion pipeline working end-to-end  
**End state:** You run `python scripts/ingest_data.py` → PDFs get chunked, embedded, and stored in FAISS  

---

## Step 1: Create the Project Skeleton

Create the folder structure. Don't write any code yet — just empty files and folders.

```
c:\ML_DL_Projects\ContexFlow\
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── generator.py
│   │   └── query_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── embeddings.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── document_store.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunking.py
│   │   ├── embedding_pipeline.py
│   │   └── metadata_extractor.py
│   └── utils/
│       ├── __init__.py
│       ├── text_cleaning.py
│       └── helpers.py
│
├── data/
│   ├── raw/            ← put your test PDFs here
│   ├── processed/
│   └── embeddings/     ← FAISS index will be saved here
│
├── frontend/
│   └── app.py
│
├── tests/
│   ├── test_rag.py
│   └── test_api.py
│
├── scripts/
│   ├── ingest_data.py
│   └── evaluate.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

### What to do:
- Create all these folders and empty `.py` files
- Add `__init__.py` to every Python package folder (can be empty)
- Drop 2-3 PDFs you're familiar with into `data/raw/` (use docs where you know the content — makes debugging easier)

### `.gitignore` — add this now:
```
.env
__pycache__/
*.pyc
data/embeddings/
data/processed/
.venv/
```

> [!TIP]
> `git init` and make your first commit here. Message: "Initial project structure"

---

## Step 2: Setup Environment & Config

### 2a. Create a virtual environment

```bash
cd c:\ML_DL_Projects\ContexFlow
python -m venv .venv
.venv\Scripts\activate
```

### 2b. `requirements.txt`

```
fastapi==0.115.0
uvicorn==0.30.0
python-dotenv==1.0.1
pydantic-settings==2.5.0
PyMuPDF==1.24.0
sentence-transformers==3.0.0
faiss-cpu==1.8.0
rank-bm25==0.2.2
transformers==4.44.0
openai==1.45.0
streamlit==1.38.0
```

```bash
pip install -r requirements.txt
```

### 2c. `.env` file

```env
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
DATA_DIR=data
```

### 2d. `app/core/config.py` — Your first real code

**What this file should do:**
- Use `pydantic-settings` to load `.env` values into a typed config class
- Create a single `Settings` instance that the rest of the app imports

**What you need to define:**
```
class Settings(BaseSettings):
    - openai_api_key: str
    - embedding_model: str (default "all-MiniLM-L6-v2")
    - chunk_size: int (default 512)
    - chunk_overlap: int (default 50)
    - top_k: int (default 5)
    - data_dir: str (default "data")
    
    class Config:
        env_file = ".env"
```

**Imports you'll need:** `from pydantic_settings import BaseSettings`

**At the bottom:** create `settings = Settings()` so other files can do `from app.core.config import settings`

> [!TIP]
> Test it! Create a quick test script: `from app.core.config import settings; print(settings.chunk_size)` — should print `512`.

---

## Step 3: Document Loader (`app/ingestion/loader.py`)

**What this file should do:**
- Load PDF files and extract text **per page**
- Return a list of "documents" where each document has: `text`, `metadata` (filename, page number)

**Think about the data structure:**
```python
# Each "document" could be a dict or a dataclass:
{
    "text": "The extracted text from page 3...",
    "metadata": {
        "source": "traffic_paper.pdf",
        "page": 3
    }
}
```

**Functions to write:**

### `load_pdf(file_path: str) -> list[dict]`
- Open PDF with PyMuPDF (`import fitz`)
- Loop through each page
- Extract text from each page
- Return list of dicts with text + metadata

### `load_directory(dir_path: str) -> list[dict]`
- Scan `data/raw/` for all `.pdf` files
- Call `load_pdf()` on each
- Return combined list of all documents

**Key PyMuPDF pattern:**
```python
import fitz  # this is PyMuPDF's import name

doc = fitz.open(file_path)
for page_num, page in enumerate(doc):
    text = page.get_text()
    # ... build your document dict
```

**Edge cases to handle:**
- Empty pages (skip them)
- Non-PDF files in the directory (skip or warn)

> [!TIP]
> **Test it!** After writing this, run:
> ```python
> from app.ingestion.loader import load_directory
> docs = load_directory("data/raw")
> print(f"Loaded {len(docs)} pages")
> print(docs[0]["text"][:200])  # first 200 chars of first page
> print(docs[0]["metadata"])
> ```
> You should see actual text from your PDF.

---

## Step 4: Text Cleaning (`app/utils/text_cleaning.py`)

**What this file should do:**
- Clean raw extracted text (PDFs are messy!)

**Function to write:**

### `clean_text(text: str) -> str`
- Remove excessive whitespace (multiple spaces → single space)
- Remove excessive newlines (multiple `\n` → single `\n`)
- Strip leading/trailing whitespace
- Optionally: remove headers/footers patterns, page numbers

**Keep it simple for now.** You can always make it smarter later.

```
Input:  "  The   quick\n\n\n\nbrown    fox  "
Output: "The quick\nbrown fox"
```

> [!TIP]
> Use Python's `re` module. `re.sub(r'\s+', ' ', text)` handles most whitespace issues.

---

## Step 5: Text Chunking (`app/ingestion/chunking.py`)

**This is one of the most important files in the entire project.**

**What this file should do:**
- Take a cleaned document text and split it into overlapping chunks
- Each chunk should be a manageable size (default: 512 characters)
- Chunks should overlap (default: 50 characters) to preserve context at boundaries

**Function to write:**

### `chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]`

**The algorithm (write it yourself, it's simple):**
1. Start at position 0
2. Take `chunk_size` characters → that's one chunk
3. Move forward by `chunk_size - chunk_overlap` characters
4. Repeat until you reach the end
5. Return list of chunk strings

**Visual example** (chunk_size=10, overlap=3):
```
Text: "ABCDEFGHIJKLMNOPQRST"

Chunk 1: "ABCDEFGHIJ"     (pos 0-9)
Chunk 2: "HIJKLMNOPQ"     (pos 7-16, overlaps "HIJ")
Chunk 3: "OPQRST"         (pos 14-19, overlaps "OPQ")
```

### `chunk_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]`

This wraps the above function but **preserves metadata**:
- Takes the list of documents from the loader
- Chunks each document's text
- Returns a NEW list where each item is:
```python
{
    "text": "chunk text here...",
    "metadata": {
        "source": "traffic_paper.pdf",
        "page": 3,
        "chunk_index": 0  # which chunk of this page
    }
}
```

> [!WARNING]
> **Don't split mid-word.** After taking your chunk, try to extend or shrink it to the nearest sentence boundary (period, newline). This is optional but improves quality.

> [!TIP]
> **Test it!** 
> ```python
> chunks = chunk_text("A" * 1000, chunk_size=512, chunk_overlap=50)
> print(f"Number of chunks: {len(chunks)}")
> print(f"Chunk sizes: {[len(c) for c in chunks]}")
> # Should see ~3 chunks, each ~512 chars, last one smaller
> ```

---

## Step 6: Embedding Model Wrapper (`app/models/embeddings.py`)

**What this file should do:**
- Load the `all-MiniLM-L6-v2` model (once!)
- Provide a function to embed text(s) into vectors

**Key design choice:** Load the model as a **singleton** so it's not re-loaded every call.

**What you need:**

### `class EmbeddingModel:`

```
__init__(self, model_name: str):
    - Load SentenceTransformer model
    - Store it as self.model

embed(self, texts: list[str]) -> numpy array:
    - self.model.encode(texts)
    - Return the embeddings
    
embed_query(self, query: str) -> numpy array:
    - Convenience method for single text
    - Return self.embed([query])[0]
```

**Import:** `from sentence_transformers import SentenceTransformer`

**Key pattern:**
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["text1", "text2"])  # returns numpy array
print(embeddings.shape)  # (2, 384) — 384-dim vectors
```

**At the bottom of the file**, create a default instance:
```python
embedding_model = EmbeddingModel(settings.embedding_model)
```

> [!TIP]
> **Test it!**
> ```python
> from app.models.embeddings import embedding_model
> vec = embedding_model.embed_query("What is traffic optimization?")
> print(vec.shape)   # should be (384,)
> print(vec[:5])     # first 5 values — should be floats
> ```
> First run will download the model (~80MB). After that, it's cached.

---

## Step 7: FAISS Vector Store (`app/db/vector_store.py`)

**This is your database.** It stores embeddings and lets you search them.

**What this file should do:**
- Create/load a FAISS index
- Add embeddings with metadata
- Search by vector
- Save/load index to/from disk

### `class VectorStore:`

```
__init__(self, dimension: int = 384):
    - Create a FAISS index (faiss.IndexFlatL2 — start simple)
    - Initialize empty list for metadata storage
    - self.index = faiss.IndexFlatL2(dimension)
    - self.metadata = []  # stores metadata for each vector

add(self, embeddings: numpy array, metadata_list: list[dict]):
    - Add embeddings to FAISS index
    - Append metadata to self.metadata list
    - IMPORTANT: metadata order must match embedding order

search(self, query_embedding: numpy array, top_k: int) -> list[dict]:
    - Search FAISS for top_k nearest neighbors
    - Return list of {text, metadata, score} dicts

save(self, directory: str):
    - Save FAISS index to disk: faiss.write_index(self.index, path)
    - Save metadata list as JSON or pickle

load(self, directory: str):
    - Load FAISS index from disk: faiss.read_index(path)
    - Load metadata from JSON or pickle
```

**Key FAISS patterns:**
```python
import faiss
import numpy as np

# Create index (384 = dimension of MiniLM embeddings)
index = faiss.IndexFlatL2(384)

# Add vectors (must be float32 numpy array)
vectors = np.array(embeddings, dtype=np.float32)
index.add(vectors)

# Search
query_vec = np.array([query_embedding], dtype=np.float32)
distances, indices = index.search(query_vec, k=5)
# distances = similarity scores
# indices = positions in the index → use to look up metadata
```

> [!IMPORTANT]
> **The metadata trick:** FAISS only stores vectors, not your text/metadata. You need to maintain a parallel list where `metadata[i]` corresponds to the `i-th` vector in the FAISS index. Save both the index AND the metadata list.

> [!TIP]
> **Test it!**
> ```python
> import numpy as np
> from app.db.vector_store import VectorStore
> 
> store = VectorStore(dimension=384)
> # Add 3 fake embeddings
> fake_vecs = np.random.rand(3, 384).astype('float32')
> fake_meta = [
>     {"text": "chunk 1", "source": "test.pdf", "page": 1},
>     {"text": "chunk 2", "source": "test.pdf", "page": 2},
>     {"text": "chunk 3", "source": "test.pdf", "page": 3},
> ]
> store.add(fake_vecs, fake_meta)
> 
> # Search
> query = np.random.rand(384).astype('float32')
> results = store.search(query, top_k=2)
> print(results)  # should return 2 results with metadata
> ```

---

## Step 8: Ingestion Pipeline (`app/ingestion/embedding_pipeline.py`)

**This is the orchestrator for ingestion.** It ties everything together.

**What this file should do:**
- Load documents → clean → chunk → embed → store in FAISS

### `class IngestionPipeline:`

```
__init__(self):
    - Initialize EmbeddingModel
    - Initialize VectorStore

run(self, data_dir: str):
    1. Load all documents from data_dir/raw/
       → use loader.load_directory()
    
    2. Clean each document's text
       → use text_cleaning.clean_text()
    
    3. Chunk all documents
       → use chunking.chunk_documents()
       → print how many chunks were created
    
    4. Extract texts from chunks for embedding
       → texts = [chunk["text"] for chunk in chunks]
    
    5. Generate embeddings
       → embeddings = self.embedding_model.embed(texts)
       → print shape to verify
    
    6. Add to vector store
       → self.vector_store.add(embeddings, chunks)
    
    7. Save vector store to disk
       → self.vector_store.save(data_dir + "/embeddings")
    
    8. Print summary:
       → "Ingested X documents, Y pages, Z chunks"
```

**The flow visualized:**
```
PDFs in data/raw/
       ↓
   loader.py        →  list of {text, metadata}
       ↓
   text_cleaning.py →  cleaned text
       ↓
   chunking.py      →  list of {chunk_text, metadata}
       ↓
   embeddings.py    →  numpy array of vectors
       ↓
   vector_store.py  →  saved FAISS index + metadata
```

---

## Step 9: Ingestion Script (`scripts/ingest_data.py`)

**This is the CLI entry point.** Very simple.

```python
# This file should:
# 1. Import the IngestionPipeline
# 2. Import settings for data_dir
# 3. Call pipeline.run(settings.data_dir)
# 4. Print success/failure message

# Hint: you'll need to handle the Python path
# Add this at the top:
import sys
sys.path.insert(0, ".")  # so "app" package is importable
```

**How it should work:**
```bash
cd c:\ML_DL_Projects\ContexFlow
python scripts/ingest_data.py
```

**Expected output:**
```
Loading documents from data/raw...
Loaded 3 PDFs, 47 pages total
Cleaning text...
Chunking documents (size=512, overlap=50)...
Created 156 chunks
Generating embeddings...
Embeddings shape: (156, 384)
Saving to data/embeddings/...
✅ Ingestion complete! 3 documents → 156 chunks stored.
```

---

## Step 10: Verify Everything Works

### Checklist:
- [ ] `python -c "from app.core.config import settings; print(settings.chunk_size)"` → prints 512
- [ ] `python -c "from app.ingestion.loader import load_directory; print(len(load_directory('data/raw')))"` → prints number of pages
- [ ] `python -c "from app.models.embeddings import embedding_model; print(embedding_model.embed_query('test').shape)"` → prints (384,)
- [ ] `python scripts/ingest_data.py` → runs without errors, creates files in `data/embeddings/`
- [ ] `data/embeddings/` contains your saved FAISS index and metadata file

### Quick retrieval test (to prove it actually works):
```python
# Run this after ingestion
import sys
sys.path.insert(0, ".")
from app.db.vector_store import VectorStore
from app.models.embeddings import embedding_model

store = VectorStore()
store.load("data/embeddings")

query = "What is the main topic?"  # adjust to your documents
query_vec = embedding_model.embed_query(query)
results = store.search(query_vec, top_k=3)

for r in results:
    print(f"Source: {r['metadata']['source']}, Page: {r['metadata']['page']}")
    print(f"Text: {r['text'][:150]}...")
    print("---")
```

If this returns relevant chunks from your PDFs → **Phase 1 is complete.** 🎉

---

## 🔑 Summary: Files You're Writing in Phase 1

| # | File | Purpose | Difficulty |
|---|---|---|---|
| 1 | `app/core/config.py` | Load env variables | Easy |
| 2 | `app/ingestion/loader.py` | Extract text from PDFs | Easy |
| 3 | `app/utils/text_cleaning.py` | Clean messy PDF text | Easy |
| 4 | `app/ingestion/chunking.py` | Split text into chunks | Medium |
| 5 | `app/models/embeddings.py` | Embed text into vectors | Easy |
| 6 | `app/db/vector_store.py` | FAISS index wrapper | Medium |
| 7 | `app/ingestion/embedding_pipeline.py` | Orchestrate ingestion | Easy |
| 8 | `scripts/ingest_data.py` | CLI entry point | Easy |

**Build them in this exact order.** Each file depends only on the ones before it.

---

> [!IMPORTANT]
> **The golden rule:** Test each file IMMEDIATELY after writing it. Don't write 5 files and then try to debug them all at once. Write one → test → commit → next.
