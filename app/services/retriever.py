from app.db.vector_store import VectorStore
from app.models.embeddings import embedding_model
from app.core.config import settings


class Retriever:
    """Retrieves relevant chunks from the vector store for a given query."""

    def __init__(self):
        self.vector_store = VectorStore()
        # load the saved FAISS index from disk
        embeddings_dir = f"{settings.data_dir}/embeddings"
        self.vector_store.load(embeddings_dir)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Find the top-K most relevant chunks for a query.
        
        Args:
            query: user's question as a string
            top_k: how many chunks to return (default from config)
            
        Returns:
            list of dicts with keys: text, metadata, score
        """
        top_k = top_k or settings.top_k

        # step 1: convert query text → vector
        query_embedding = embedding_model.embed_query(query)

        # step 2: search FAISS for nearest vectors
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results
