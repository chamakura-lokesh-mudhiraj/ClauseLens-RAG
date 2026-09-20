from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVALUATION_DATA_DIR = PROJECT_ROOT / "data" / "evaluation"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval settings
TOP_K = 10

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50