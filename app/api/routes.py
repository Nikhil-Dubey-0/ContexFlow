from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

# initialize pipeline once — reused across all requests
pipeline = RAGPipeline()


def _has_documents() -> bool:
    """True if any documents have been indexed."""
    return bool(getattr(pipeline.retriever, "documents", []))


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question and get an answer with sources."""
    if not _has_documents():
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed yet. Ingest documents first.",
        )

    try:
        result = pipeline.query(
            request.question,
            top_k=request.top_k,
            chat_history=[m.model_dump() for m in request.chat_history],
        )
    except Exception as e:
        # upstream (LLM/embedding) failure — surface as a gateway error
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    return result


@router.post("/query/stream")
def query_stream(request: QueryRequest):
    """Same as /query but streams the answer token-by-token (text/plain)."""
    if not _has_documents():
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed yet. Ingest documents first.",
        )

    try:
        token_stream, _sources, _rewritten = pipeline.stream_query(
            request.question,
            top_k=request.top_k,
            chat_history=[m.model_dump() for m in request.chat_history],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    return StreamingResponse(token_stream, media_type="text/plain")


@router.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "documents_indexed": _has_documents()}
