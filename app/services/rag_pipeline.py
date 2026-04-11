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

    def query(self, question: str, top_k: int = None, 
              chat_history: list[dict] = None) -> dict:
        """Process a user question end-to-end.
        
        Flow:
        1. Rewrite query for better retrieval
        2. Retrieve chunks (FAISS + BM25 hybrid)
        3. Rerank chunks with cross-encoder
        4. Generate answer from top chunks + chat history
        """
        # step 1: rewrite query for better search
        rewritten_query = self.query_processor.rewrite(question)

        # step 2: hybrid retrieval
        retrieved_chunks = self.retriever.retrieve(rewritten_query, top_k=(top_k or 5) * 2)

        # step 3: rerank
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k or 5)

        # step 4: generate with history
        result = self.generator.generate(question, reranked_chunks, chat_history=chat_history)
        
        result["query"] = question
        result["rewritten_query"] = rewritten_query

        return result

    def stream_query(self, question: str, top_k: int = None,
                     chat_history: list[dict] = None):
        """Same as query() but streams the answer.
        
        Returns:
            tuple: (token_generator, sources, rewritten_query)
        """
        # steps 1-3 are the same (not streamed — they're fast)
        rewritten_query = self.query_processor.rewrite(question)
        retrieved_chunks = self.retriever.retrieve(rewritten_query, top_k=(top_k or 5) * 2)
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k or 5)

        # step 4: stream the generation
        token_stream = self.generator.stream_generate(question, reranked_chunks, chat_history)
        
        # build sources list
        sources = []
        for chunk in reranked_chunks:
            sources.append({
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "snippet": chunk["text"][:150] + "..."
            })

        return token_stream, sources, rewritten_query
