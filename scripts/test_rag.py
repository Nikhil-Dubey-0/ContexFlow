import sys
sys.path.insert(0, ".")

from app.services.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

query = "what is the unit 3 of deep learning syllabus?"
print(f"\n🔍 Query: {query}\n")

result = pipeline.query(query, top_k=5)

print("=" * 50)
print(f"🔄 Rewritten query: {result.get('rewritten_query', 'N/A')}")
print(f"\n📝 ANSWER:\n{result['answer']}")
print("=" * 50)
print("\n📚 SOURCES:")
for s in result["sources"]:
    print(f"  - {s['source']} (Page {s['page']})")
