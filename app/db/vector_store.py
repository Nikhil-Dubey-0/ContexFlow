import os
import json                    # to save/load metadata as JSON
import numpy as np
import faiss                   # Facebook AI Similarity Search


class VectorStore:
    """FAISS-based vector store with metadata tracking."""

    def __init__(self, dimension: int = 384):
        # IndexFlatL2 = brute-force search using L2 (Euclidean) distance
        # simple, accurate, good enough for <100k vectors
        # for millions of vectors, you'd use IndexIVFFlat (approximate search)
        self.index = faiss.IndexFlatL2(dimension)
        
        # parallel list — metadata[i] corresponds to the i-th vector in the index
        self.metadata = []
        
        self.dimension = dimension

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]):
        """Add embeddings and their metadata to the store.
        
        Args:
            embeddings: numpy array of shape (n, 384)
            metadata_list: list of dicts, one per embedding
        """
        # FAISS requires float32 — enforce it
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # sanity check: number of vectors must match number of metadata entries
        assert len(embeddings) == len(metadata_list), \
            f"Mismatch: {len(embeddings)} embeddings but {len(metadata_list)} metadata entries"
        
        # add vectors to FAISS index
        self.index.add(embeddings)
        
        # add metadata to our parallel list
        self.metadata.extend(metadata_list)
        
        print(f"📥 Added {len(embeddings)} vectors. Total in store: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for the top-k most similar vectors.
        
        Args:
            query_embedding: numpy array of shape (384,) — single query vector
            top_k: how many results to return
            
        Returns:
            list of dicts with keys: text, metadata, score
        """
        # FAISS expects a 2D array — reshape (384,) → (1, 384)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # search returns:
        #   distances: array of shape (1, top_k) — L2 distances (lower = more similar)
        #   indices: array of shape (1, top_k) — positions in the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):       # indices[0] because we have 1 query
            if idx == -1:                           # FAISS returns -1 if fewer than top_k results exist
                continue
            results.append({
                "text": self.metadata[idx]["text"],
                "metadata": self.metadata[idx]["metadata"],
                "score": float(distances[0][i])     # convert numpy float to Python float
            })
        
        return results

    def save(self, directory: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)      # create directory if it doesn't exist
        
        # save the FAISS index — binary file
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # save metadata as JSON
        meta_path = os.path.join(directory, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        
        print(f"💾 Saved {self.index.ntotal} vectors to {directory}")

    def load(self, directory: str):
        """Load FAISS index and metadata from disk."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.json")
        
        # check files exist before loading
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No FAISS index found at {index_path}")
        
        # load FAISS index
        self.index = faiss.read_index(index_path)
        
        # load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        print(f"📂 Loaded {self.index.ntotal} vectors from {directory}")
