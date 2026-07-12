from typing import Literal
from pydantic import BaseModel, Field  # Pydantic validates request/response data


class ChatMessage(BaseModel):
    """One prior turn of the conversation."""
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """What the user sends to /query endpoint."""
    # the user's question — bounded to avoid abuse / oversized prompts
    question: str = Field(..., min_length=1, max_length=2000)
    # optional: how many chunks to retrieve (kept in a sane range)
    top_k: int = Field(5, ge=1, le=20)
    # optional prior turns so the API supports multi-turn conversations
    chat_history: list[ChatMessage] = Field(default_factory=list)


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
