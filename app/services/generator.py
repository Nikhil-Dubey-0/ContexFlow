from openai import OpenAI
from app.core.config import settings


class Generator:
    """Generates answers using Groq LLM with retrieved context."""

    def __init__(self):
        # Groq uses OpenAI-compatible API — just different base URL
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1"  # point to Groq instead of OpenAI
        )
        self.model = settings.llm_model
        print(f"✅ Generator initialized with {self.model}")

    def generate(self, query: str, context_chunks: list[dict]) -> dict:
        """Generate an answer from query + retrieved chunks."""
        
        # --- Build context from retrieved chunks ---
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['text']}")
        
        context_string = "\n\n---\n\n".join(context_parts)

        # --- System prompt (instructions for the LLM) ---
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer the question using ONLY the information from the context below.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Cite your sources by referencing the [Source X] tags.
4. Be concise but thorough.
5. Do not make up information that is not in the context."""

        # --- User message with context + question ---
        user_message = f"""CONTEXT:
{context_string}

QUESTION: {query}"""

        # --- Call Groq (OpenAI-compatible format) ---
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,       # low = more factual, less creative
                max_tokens=1024        # max length of the answer
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response: {e}"

        # --- Build sources list ---
        sources = []
        for chunk in context_chunks:
            sources.append({
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "snippet": chunk["text"][:150] + "..."
            })

        return {
            "answer": answer,
            "sources": sources
        }
