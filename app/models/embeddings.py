from sentence_transformers import SentenceTransformer
import numpy as np
from app.core.config import settings


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating embeddings."""

    def __init__(self, model_name: str = None):
        # use model name from config if not explicitly provided
        model_name = model_name or settings.embedding_model

        print(f"🔄 Loading embedding model: {model_name}...")
        # this downloads the model on first run (~80MB), then uses cached version
        self.model = SentenceTransformer(model_name)
        print(f"✅ Embedding model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.
        
        Args:
            texts: list of strings to embed
            
        Returns:
            numpy array of shape (len(texts), 384)
        """
        # show_progress_bar=True gives you a nice progress bar for large batches
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings, dtype=np.float32)  # FAISS requires float32

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.
        
        Returns:
            numpy array of shape (384,)
        """
        # encode() can take a single string too, returns 1D array
        return self.model.encode(query).astype(np.float32)


# --- Singleton instance ---
# create ONE instance so the model is loaded once and reused everywhere
# other files do: from app.models.embeddings import embedding_model
embedding_model = EmbeddingModel()
