import sys
# add project root to Python path so "app" package is importable
sys.path.insert(0, ".")

# TLS-via-OS-store + UTF-8 console; before any app module is imported
from app.bootstrap import init as _init
_init()

from app.ingestion.embedding_pipeline import IngestionPipeline


if __name__ == "__main__":
    # pass --force to rebuild the whole index from scratch (e.g. after the
    # embedding/index format changes); default is incremental
    force = "--force" in sys.argv

    pipeline = IngestionPipeline()
    pipeline.run(force=force)
