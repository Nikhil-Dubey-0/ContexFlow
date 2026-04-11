from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="ContexFlow",
    description="RAG system for document Q&A",
    version="1.0.0"
)

# register all routes
app.include_router(router)
