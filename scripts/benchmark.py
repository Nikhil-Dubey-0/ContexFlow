"""Quick benchmark to get resume-worthy metrics for ContexFlow."""
import sys, time
sys.path.insert(0, ".")

from app.models.embeddings import embedding_model
from app.db.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.reranker import Reranker
from app.services.query_processor import QueryProcessor
from app.services.generator import Generator
from app.core.config import settings
import numpy as np

print("=" * 60)
print("ContexFlow Benchmark")
print("=" * 60)

# --- Load components ---
retriever = Retriever()
reranker = Reranker()
query_processor = QueryProcessor()
generator = Generator()

doc_count = len(retriever.documents)
print(f"\n📊 Indexed chunks: {doc_count}")

# --- Test queries ---
test_queries = [
    "Semester III subjects list",
    "What is Edge AI",
    "What projects has Nikhil worked on",
    "Semester 6 subjects",
    "What are advantages of edge ai over cloud ai",
]

# --- Benchmark 1: FAISS-only vs Hybrid retrieval quality ---
print("\n" + "=" * 60)
print("🔍 FAISS-only vs Hybrid Retrieval Quality")
print("=" * 60)

# ground truth: which pages should be in top-5 for each query
ground_truth = {
    "Semester III subjects list": [6],      # page 6 has semester III
    "Semester 6 subjects": [7],             # page 7 has semester VI
}

faiss_hits = 0
hybrid_hits = 0
total_checks = 0

for query, expected_pages in ground_truth.items():
    # FAISS only
    query_emb = embedding_model.embed_query(query)
    faiss_results = retriever.vector_store.search(query_emb, top_k=5)
    faiss_pages = [r["metadata"]["page"] for r in faiss_results]
    
    # Hybrid (FAISS + BM25 + RRF)
    hybrid_results = retriever.retrieve(query, top_k=5)
    hybrid_pages = [r["metadata"]["page"] for r in hybrid_results]
    
    for page in expected_pages:
        total_checks += 1
        if page in faiss_pages:
            faiss_hits += 1
        if page in hybrid_pages:
            hybrid_hits += 1
    
    print(f"\nQuery: '{query}'")
    print(f"  Expected page(s): {expected_pages}")
    print(f"  FAISS top-5 pages: {faiss_pages} → {'✅ HIT' if any(p in faiss_pages for p in expected_pages) else '❌ MISS'}")
    print(f"  Hybrid top-5 pages: {hybrid_pages} → {'✅ HIT' if any(p in hybrid_pages for p in expected_pages) else '❌ MISS'}")

faiss_recall = faiss_hits / total_checks * 100
hybrid_recall = hybrid_hits / total_checks * 100
print(f"\n📊 FAISS-only recall@5: {faiss_recall:.0f}%")
print(f"📊 Hybrid recall@5: {hybrid_recall:.0f}%")
print(f"📊 Improvement: +{hybrid_recall - faiss_recall:.0f}%")

# --- Benchmark 2: Latency ---
print("\n" + "=" * 60)
print("⏱️ Latency Benchmark (average over queries)")
print("=" * 60)

retrieval_times = []
rerank_times = []
rewrite_times = []
total_times = []

for query in test_queries:
    # query rewrite
    t0 = time.time()
    rewritten = query_processor.rewrite(query)
    rewrite_time = time.time() - t0
    rewrite_times.append(rewrite_time)
    
    # retrieval
    t1 = time.time()
    chunks = retriever.retrieve(rewritten, top_k=10)
    retrieval_time = time.time() - t1
    retrieval_times.append(retrieval_time)
    
    # reranking
    t2 = time.time()
    reranked = reranker.rerank(query, chunks, top_k=5)
    rerank_time = time.time() - t2
    rerank_times.append(rerank_time)
    
    total = rewrite_time + retrieval_time + rerank_time
    total_times.append(total)
    
    print(f"  '{query[:40]}...' → rewrite: {rewrite_time:.2f}s, retrieve: {retrieval_time:.3f}s, rerank: {rerank_time:.3f}s")

print(f"\n📊 Avg query rewrite: {np.mean(rewrite_times):.2f}s")
print(f"📊 Avg retrieval (hybrid): {np.mean(retrieval_times)*1000:.0f}ms")
print(f"📊 Avg reranking: {np.mean(rerank_times)*1000:.0f}ms")
print(f"📊 Avg total pipeline (excl. generation): {np.mean(total_times):.2f}s")

print("\n" + "=" * 60)
print("✅ Benchmark complete!")
print("=" * 60)
