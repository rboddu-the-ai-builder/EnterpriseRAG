# embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model ready.")
    return embeddings