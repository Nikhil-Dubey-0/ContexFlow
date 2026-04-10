import os
from app.ingestion.loader import load_directory          # load PDFs
from app.ingestion.chunking import chunk_documents       # split into chunks
from app.models.embeddings import embedding_model        # embed text → vectors
from app.db.vector_store import VectorStore              # store in FAISS
from app.core.config import settings                     # config values


class IngestionPipeline:
    """Orchestrates the full ingestion: PDFs → chunks → embeddings → FAISS."""

    def __init__(self):
        self.vector_store = VectorStore()

    def run(self, data_dir: str = None):
        """Run the full ingestion pipeline.
        
        Args:
            data_dir: path to data directory (contains raw/ and embeddings/ subdirs)
        """
        data_dir = data_dir or settings.data_dir
        raw_dir = os.path.join(data_dir, "raw")
        embeddings_dir = os.path.join(data_dir, "embeddings")

        # --- Step 1: Load documents ---
        print("\n📄 Step 1: Loading documents...")
        documents = load_directory(raw_dir)
        if not documents:
            print("❌ No documents to process. Exiting.")
            return

        # --- Step 2: Chunk documents (cleaning happens inside chunk_documents) ---
        print("\n✂️ Step 2: Chunking documents...")
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        print(f"   Created {len(chunks)} chunks")

        # --- Step 3: Generate embeddings ---
        print("\n🧠 Step 3: Generating embeddings...")
        # extract just the text from each chunk for embedding
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.embed(texts)
        print(f"   Embeddings shape: {embeddings.shape}")

        # --- Step 4: Store in FAISS ---
        print("\n💾 Step 4: Storing in vector database...")
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save(embeddings_dir)

        # --- Summary ---
        print(f"\n{'='*50}")
        print(f"✅ Ingestion complete!")
        print(f"   Documents: {len(set(d['metadata']['source'] for d in documents))}")
        print(f"   Pages: {len(documents)}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Embeddings: {embeddings.shape}")
        print(f"   Saved to: {embeddings_dir}")
        print(f"{'='*50}")
