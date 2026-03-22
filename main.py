# main.py
from embeddings import load_embedding_model
from ingest import ingest_documents, wipe_vectorstore

WIPE_AND_REBUILD = False  # ← change to True when you want to wipe


def main():
    if WIPE_AND_REBUILD:
        wipe_vectorstore()    
    # Step 1: load embedding model
    embeddings = load_embedding_model()

    # Step 2: ingest documents (handles all logic automatically)
    vectorstore = ingest_documents(embeddings)

    if vectorstore is None:
        print("No documents in vector store yet. Add PDFs to /docs and run again.")
        return

    # Step 3: quick test query to confirm everything works
    print("\nRunning test query...")
    results = vectorstore.similarity_search("What is this document about?", k=3)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Page:   {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content[:200])

if __name__ == "__main__":
    main()