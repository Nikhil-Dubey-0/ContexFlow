from openai import OpenAI
from app.core.config import settings


class QueryProcessor:
    """Rewrites user queries for better retrieval."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def rewrite(self, query: str) -> str:
        """Rewrite a user query to be more specific and search-friendly.
        
        Example:
            'what does nikhil know' → 'What are Nikhil's technical skills, 
             programming languages, tools, and professional experience?'
        """
        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": 
                     "Rewrite the following user question to be more specific and search-friendly. "
                     "Expand abbreviations, add relevant synonyms, and make it clearer. "
                     "Return ONLY the rewritten query, nothing else."},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,       # deterministic — same input = same output
                max_tokens=100         # queries should be short
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"🔄 Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            # if rewriting fails, just use the original query — don't crash
            print(f"⚠️ Query rewrite failed: {e}. Using original query.")
            return query
