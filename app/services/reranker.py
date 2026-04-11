from sentence_transformers import CrossEncoder  # deeper model that reads query+doc pairs


class Reranker:
    """Re-ranks retrieved chunks using a cross-encoder model.
    
    Why? FAISS/BM25 rank independently — they score each chunk without 
    considering the query deeply. A cross-encoder reads (query, chunk) 
    as a PAIR and gives a much more accurate relevance score.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"🔄 Loading reranker model: {model_name}...")
        # CrossEncoder is ~80MB, runs on CPU, loads once
        self.model = CrossEncoder(model_name)
        print(f"✅ Reranker loaded!")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Re-rank chunks by relevance to the query.
        
        Args:
            query: the user's question
            chunks: list of dicts from retriever (each has text, metadata, score)
            top_k: how many to return after reranking
            
        Returns:
            reranked list of chunks (best first)
        """
        if not chunks:
            return []

        # create (query, chunk_text) pairs for the cross-encoder
        # it reads both together and scores how relevant the chunk is to the query
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # get relevance scores — higher = more relevant
        scores = self.model.predict(pairs)

        # attach scores to chunks
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])

        # sort by rerank score (highest first)
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
