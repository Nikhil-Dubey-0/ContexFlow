from pydantic import BaseModel  # Pydantic validates request/response data


class QueryRequest(BaseModel):
    """What the user sends to /query endpoint."""
    question: str               # the user's question
    top_k: int = 5              # optional: how many chunks to retrieve


class SourceInfo(BaseModel):
    """One source reference in the response."""
    source: str                 # filename
    page: int                   # page number
    snippet: str                # text preview


class QueryResponse(BaseModel):
    """What the API returns."""
    query: str                  # echo back the original question
    answer: str                 # the LLM's answer
    sources: list[SourceInfo]   # list of source references
