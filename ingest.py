# ingest.py
import os
import json
import hashlib
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import (
    CHROMA_DIR, DOCS_DIR, REGISTRY_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# ── Splitter ────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ── Registry helpers ─────────────────────────────────────────
def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        return json.load(open(REGISTRY_FILE))
    return {}

def save_registry(registry):
    json.dump(registry, open(REGISTRY_FILE, "w"), indent=2)

# ── Load or create ChromaDB ──────────────────────────────────
def load_or_create_vectorstore(embeddings):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Existing vector store found — loading from disk.")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        print("No vector store found — will create after ingestion.")
        return None

# ── Check and embed only new files ───────────────────────────
def ingest_documents(embeddings):
    registry = load_registry()
    vectorstore = load_or_create_vectorstore(embeddings)
    new_chunks = []

    # Check every PDF in /docs
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(DOCS_DIR, filename)
        file_hash = get_file_hash(filepath)

        if file_hash in registry:
            print(f"Skipping '{filename}' — already embedded.")
            continue

        # New file — load and chunk it
        print(f"New file detected: '{filename}' — loading and chunking...")
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        new_chunks.extend(chunks)

        # Record in registry
        registry[file_hash] = {
            "filename": filename,
            "chunks": len(chunks)
        }
        print(f"  → {len(chunks)} chunks created.")

    # If no new files found at all
    if not new_chunks:
        print("No new files to embed.")
        return vectorstore

    # Add to existing store or create fresh one
    if vectorstore is not None:
        print(f"Adding {len(new_chunks)} new chunks to existing vector store...")
        vectorstore.add_documents(new_chunks)
    else:
        print(f"Creating new vector store with {len(new_chunks)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

    save_registry(registry)
    print("Ingestion complete.")
    return vectorstore

# ── Wipe everything ──────────────────────────────────────────
def wipe_vectorstore():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Vector store wiped.")
    if os.path.exists(REGISTRY_FILE):
        os.remove(REGISTRY_FILE)
        print("Registry wiped.")