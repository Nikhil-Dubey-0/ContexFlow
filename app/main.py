from app.bootstrap import init as _init
_init()  # TLS-via-OS-store + UTF-8; before routes import (which builds clients)

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="ContexFlow",
    description="RAG system for document Q&A",
    version="1.0.0"
)

# register all routes
app.include_router(router)
