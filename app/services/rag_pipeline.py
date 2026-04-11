from app.services.retriever import Retriever
from app.services.generator import Generator
from app.services.query_processor import QueryProcessor
from app.services.reranker import Reranker


class RAGPipeline:
    """Main orchestrator: query rewrite → retrieve → rerank → generate."""

    def __init__(self):
        self.query_processor = QueryProcessor()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()

    def query(self, question: str, top_k: int = None) -> dict:
        """Process a user question end-to-end.
        
        Flow:
        1. Rewrite query for better retrieval
        2. Retrieve chunks (FAISS + BM25 hybrid)
        3. Rerank chunks with cross-encoder
        4. Generate answer from top chunks
        """
        # step 1: rewrite query for better search
        rewritten_query = self.query_processor.rewrite(question)

        # step 2: hybrid retrieval (fetches more to give reranker material)
        retrieved_chunks = self.retriever.retrieve(rewritten_query, top_k=(top_k or 5) * 2)

        # step 3: rerank — cross-encoder picks the truly relevant ones
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k or 5)

        # step 4: generate answer using ORIGINAL question + reranked context
        result = self.generator.generate(question, reranked_chunks)
        
        # include metadata for debugging/display
        result["query"] = question
        result["rewritten_query"] = rewritten_query

        return result
