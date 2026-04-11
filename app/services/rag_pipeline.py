from app.services.retriever import Retriever
from app.services.generator import Generator


class RAGPipeline:
    """Main orchestrator: ties retriever + generator together."""

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question: str, top_k: int = None) -> dict:
        """Process a user question end-to-end.
        
        Args:
            question: the user's question
            top_k: number of chunks to retrieve
            
        Returns:
            dict with 'answer', 'sources', and 'query'
        """
        # step 1: retrieve relevant chunks
        chunks = self.retriever.retrieve(question, top_k=top_k)

        # step 2: generate answer from chunks
        result = self.generator.generate(question, chunks)

        # step 3: add the original query to the response
        result["query"] = question

        return result
