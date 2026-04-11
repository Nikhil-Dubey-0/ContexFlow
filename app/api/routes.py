from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

# initialize pipeline once — reused across all requests
pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question and get an answer with sources."""
    result = pipeline.query(request.question, top_k=request.top_k)
    return result


@router.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
