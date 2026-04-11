from openai import OpenAI
from app.core.config import settings


class Generator:
    """Generates answers using Groq LLM with retrieved context."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = settings.llm_model
        print(f"✅ Generator initialized with {self.model}")

    def generate(self, query: str, context_chunks: list[dict], 
                 chat_history: list[dict] = None) -> dict:
        """Generate an answer from query + retrieved chunks + conversation history."""
        
        # --- Build context from retrieved chunks ---
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['text']}")
        
        context_string = "\n\n---\n\n".join(context_parts)

        # --- System prompt ---
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer the question using ONLY the information from the context below.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Cite your sources by referencing the [Source X] tags.
4. Be concise but thorough.
5. Do not make up information that is not in the context."""

        # --- Build messages list ---
        messages = [{"role": "system", "content": system_prompt}]
        
        # add conversation history (last 6 messages = 3 turns)
        # this gives the LLM context about the ongoing conversation
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # add current user message with context
        user_message = f"""CONTEXT:
{context_string}

QUESTION: {query}"""
        messages.append({"role": "user", "content": user_message})

        # --- Call Groq ---
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
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

    def stream_generate(self, query: str, context_chunks: list[dict],
                        chat_history: list[dict] = None):
        """Stream an answer token by token. Yields chunks of text.
        
        Same logic as generate(), but uses stream=True.
        """
        # --- Build context ---
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['text']}")
        
        context_string = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer the question using ONLY the information from the context below.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Cite your sources by referencing the [Source X] tags.
4. Be concise but thorough.
5. Do not make up information that is not in the context."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        user_message = f"""CONTEXT:
{context_string}

QUESTION: {query}"""
        messages.append({"role": "user", "content": user_message})

        # --- Stream from Groq ---
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=True  # ← the only difference from generate()
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error generating response: {e}"
