from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"

    # Embeddings (local, unchanged)
    embedding_model: str = 'all-MiniLM-L6-v2'
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    # Paths
    data_dir: str = "data"

    class Config:
        env_file = ".env"

settings = Settings()

