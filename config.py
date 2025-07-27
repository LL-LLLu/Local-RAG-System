import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model settings
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = "text-embedding-ada-002" if USE_OPENAI_EMBEDDINGS else "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_documents"

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "openai" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")  # or "gpt-4", "claude-3-sonnet-20240229"
LLM_TEMPERATURE = 0.7
MAX_TOKENS = 500

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)