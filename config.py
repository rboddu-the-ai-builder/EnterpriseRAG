# config.py
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"
REGISTRY_FILE = "./embedded_files.json"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVAL = 10   # fetch more candidates initially
TOP_K_RERANKED = 5     # keep only top 5 after reranking

# BM25
BM25_WEIGHT = 0.5      # how much weight to give keyword search
VECTOR_WEIGHT = 0.5    # how much weight to give semantic search

# Prompt versioning
PROMPT_VERSION = "v1"
PROMPTS_DIR = "./prompts"

# Eval
EVAL_DATASET_FILE = "./data/eval_dataset.json"
FAITHFULNESS_THRESHOLD = 0.85
RELEVANCY_THRESHOLD = 0.80