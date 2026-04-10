import sys
# add project root to Python path so "app" package is importable
sys.path.insert(0, ".")

from app.ingestion.embedding_pipeline import IngestionPipeline


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
