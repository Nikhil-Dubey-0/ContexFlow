import numpy as np
from rank_bm25 import BM25Okapi            # keyword-based search
from app.db.vector_store import VectorStore
from app.models.embeddings import embedding_model
from app.core.config import settings


class Retriever:
    # retrieve relevent chunks from the vector store for a given query
    """Hybrid retriever: combines FAISS (semantic) + BM25 (keyword) search."""

    def __init__(self):
        self.vector_store = VectorStore()
        embeddings_dir = f"{settings.data_dir}/embeddings"
        
        self.documents = []
        self.bm25 = None
        
        # try to load existing index — might not exist on fresh deployment
        try:
            self.vector_store.load(embeddings_dir)
            self.documents = self.vector_store.metadata
            tokenized_docs = [doc["text"].lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"✅ BM25 index built with {len(tokenized_docs)} documents")
        except Exception:
            print("⚠️ No existing index found. Upload documents to get started.")

    def _faiss_search(self, query: str, top_k: int) -> list[dict]:
        """Semantic search using FAISS embeddings."""
        query_embedding = embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Keyword search using BM25."""
        # tokenize the query the same way we tokenized documents
        query_tokens = query.lower().split()
        
        # get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # get top-k indices (highest scores first)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include docs with non-zero relevance
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(scores[idx])
                })
        return results

    def _reciprocal_rank_fusion(self, faiss_results: list[dict], 
                                 bm25_results: list[dict], 
                                 k: int = 60) -> list[dict]:
        """Merge two ranked lists using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum of 1 / (k + rank) for each list the doc appears in.
        k=60 is the standard constant (from the original paper).
        
        Why RRF? It doesn't care about the actual scores (which are on 
        different scales for FAISS(L2 distances) vs BM25(term frequency score)). It only cares about RANK position.
        """
        # build a dict: chunk_key → {doc_data, rrf_score}
        doc_scores = {}
        
        for rank, doc in enumerate(faiss_results):
            # create a unique key for each chunk
            key = f"{doc['metadata']['source']}_{doc['metadata']['page']}_{doc['text'][:50]}"
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0}
            doc_scores[key]["score"] += 1 / (k + rank + 1)

        for rank, doc in enumerate(bm25_results):
            key = f"{doc['metadata']['source']}_{doc['metadata']['page']}_{doc['text'][:50]}"
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0}
            doc_scores[key]["score"] += 1 / (k + rank + 1)

        # sort by combined RRF score (highest first)
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # return the documents with their RRF scores
        results = []
        for item in sorted_docs:
            doc = item["doc"]
            doc["score"] = item["score"]  # replace original score with RRF score
            results.append(doc)
        
        return results

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Hybrid retrieval: FAISS + BM25, merged with RRF.
        
        Args:
            query: user's search query
            top_k: how many final results to return
        """
        if not self.documents:
            return []  # no documents indexed yet
        
        top_k = top_k or settings.top_k

        # get results from both systems (fetch more than top_k to give RRF more to work with)
        faiss_results = self._faiss_search(query, top_k=top_k * 2)
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        print(f"🔍 FAISS returned {len(faiss_results)} results")
        print(f"🔍 BM25 returned {len(bm25_results)} results")

        # merge using RRF
        merged = self._reciprocal_rank_fusion(faiss_results, bm25_results)

        # return top_k from merged results
        return merged[:top_k]
