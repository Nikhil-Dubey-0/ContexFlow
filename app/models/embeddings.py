from sentence_transformers import SentenceTransformer
import numpy as np
from app.core.config import settings


def _load_with_offline_fallback(loader, label: str):
    """Load a HuggingFace model, falling back to the local cache if the
    network is unavailable (flaky/offline environments).

    On first run the model is downloaded (~80MB); afterwards it's cached, so
    a network hiccup shouldn't take the app down. The retry passes
    local_files_only=True, which skips the online metadata check entirely
    (setting HF_HUB_OFFLINE here is too late — it's read at import time).
    """
    try:
        return loader()
    except Exception as e:
        print(f"⚠️ Online load of {label} failed ({e}); retrying from local cache...")
        return loader(local_files_only=True)


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating embeddings."""

    def __init__(self, model_name: str = None):
        # use model name from config if not explicitly provided
        model_name = model_name or settings.embedding_model

        print(f"🔄 Loading embedding model: {model_name}...")
        self.model = _load_with_offline_fallback(
            lambda **kw: SentenceTransformer(model_name, **kw),
            f"embedding model '{model_name}'",
        )
        print(f"✅ Embedding model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.

        Args:
            texts: list of strings to embed

        Returns:
            numpy array of shape (len(texts), 384), L2-normalized
        """
        # normalize_embeddings=True returns unit-length vectors so that
        # inner product == cosine similarity (what MiniLM is trained for).
        # show_progress_bar=True gives you a nice progress bar for large batches
        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)  # FAISS requires float32

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            numpy array of shape (384,), L2-normalized
        """
        # normalize so cosine similarity works against the normalized index
        return self.model.encode(query, normalize_embeddings=True).astype(np.float32)


# --- Singleton instance ---
# create ONE instance so the model is loaded once and reused everywhere
# other files do: from app.models.embeddings import embedding_model
embedding_model = EmbeddingModel()
