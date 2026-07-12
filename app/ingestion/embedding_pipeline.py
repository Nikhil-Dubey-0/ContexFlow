import os
from app.ingestion.loader import load_pdf, load_docx     # per-file loaders
from app.ingestion.chunking import chunk_documents       # split into chunks
from app.models.embeddings import embedding_model        # embed text → vectors
from app.db.vector_store import VectorStore              # store in FAISS
from app.core.config import settings                     # config values


class IngestionPipeline:
    """Orchestrates ingestion: PDFs/DOCX → chunks → embeddings → FAISS.

    Incremental by default: an existing index is loaded and only files that
    aren't already indexed get processed. Pass force=True to rebuild fully.
    """

    def __init__(self):
        self.vector_store = VectorStore()

    @staticmethod
    def _load_file(file_path: str, file_name: str) -> list[dict]:
        """Load a single supported file, or [] if unsupported."""
        lower = file_name.lower()
        if lower.endswith(".pdf"):
            return load_pdf(file_path)
        if lower.endswith(".docx"):
            return load_docx(file_path)
        return []

    def run(self, data_dir: str = None, force: bool = False):
        """Run the ingestion pipeline.

        Args:
            data_dir: path to data directory (contains raw/ and embeddings/)
            force: if True, rebuild the whole index from scratch instead of
                   appending only new files
        """
        data_dir = data_dir or settings.data_dir
        raw_dir = os.path.join(data_dir, "raw")
        embeddings_dir = os.path.join(data_dir, "embeddings")

        if not os.path.exists(raw_dir):
            print(f"❌ Raw directory not found: {raw_dir}")
            return

        # --- Step 0: Load existing index (incremental mode) ---
        indexed_sources = set()
        if not force:
            try:
                self.vector_store.load(embeddings_dir)
                indexed_sources = {
                    m["metadata"]["source"] for m in self.vector_store.metadata
                }
                print(
                    f"📂 Existing index: {self.vector_store.index.ntotal} chunks "
                    f"from {len(indexed_sources)} document(s)"
                )
            except Exception:
                print("ℹ️ No existing index — building from scratch.")

        # --- Step 1: Decide which files are new ---
        all_files = sorted(os.listdir(raw_dir))
        supported = [f for f in all_files if f.lower().endswith((".pdf", ".docx"))]
        new_files = [f for f in supported if f not in indexed_sources]

        skipped = sorted(set(supported) & indexed_sources)
        if skipped:
            print(f"⏭️ Already indexed (skipping): {', '.join(skipped)}")

        if not new_files:
            print("✅ Nothing new to ingest — index is up to date.")
            return

        # --- Step 2: Load only the new documents ---
        print(f"\n📄 Step 1: Loading {len(new_files)} new document(s)...")
        documents = []
        for file_name in new_files:
            documents.extend(self._load_file(os.path.join(raw_dir, file_name), file_name))

        if not documents:
            print("❌ New files contained no extractable text. Exiting.")
            return

        # --- Step 3: Chunk (cleaning happens inside chunk_documents) ---
        print("\n✂️ Step 2: Chunking documents...")
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        print(f"   Created {len(chunks)} chunks")

        # --- Step 4: Generate embeddings ---
        print("\n🧠 Step 3: Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.embed(texts)
        print(f"   Embeddings shape: {embeddings.shape}")

        # --- Step 5: Store in FAISS (append to whatever is already loaded) ---
        print("\n💾 Step 4: Storing in vector database...")
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save(embeddings_dir)

        # --- Summary ---
        print(f"\n{'='*50}")
        print("✅ Ingestion complete!")
        print(f"   New documents: {len(new_files)}")
        print(f"   New chunks: {len(chunks)}")
        print(f"   Total chunks in index: {self.vector_store.index.ntotal}")
        print(f"   Saved to: {embeddings_dir}")
        print(f"{'='*50}")
