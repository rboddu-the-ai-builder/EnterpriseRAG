# query.py
from embeddings import load_embedding_model
from ingest import load_or_create_vectorstore
from llm import load_llm
from rag import build_rag_chain, format_chunks
from retriever import build_bm25_index, retrieve

def run_query(question: str):
    # load everything
    embeddings = load_embedding_model()
    vectorstore = load_or_create_vectorstore(embeddings)

    if vectorstore is None:
        print("No vector store found. Run main.py first.")
        return

    llm = load_llm()

    # build BM25 index from existing ChromaDB chunks
    bm25_index, documents, metadatas = build_bm25_index(vectorstore)

    # build RAG chain
    rag_chain = build_rag_chain(llm)

    # retrieve chunks using hybrid search + reranking
    chunks = retrieve(question, vectorstore, bm25_index, documents, metadatas)

    if not chunks:
        print("No relevant chunks found.")
        return

    # run through RAG chain
    answer = rag_chain.invoke({
        "chunks": chunks,
        "question": question
    })

    # grounding checks
    refused = "cannot answer" in answer.lower()
    has_citations = "[source:" in answer.lower()

    # print results
    print(f"\nQuestion: {question}")
    print("=" * 50)
    print(f"Answer:\n{answer}")
    print("=" * 50)
    print(f"Citations present:      {has_citations}")
    print(f"Refused (out of scope): {refused}")
    print(f"Chunks retrieved:       {len(chunks)}")
    print("\nSources used:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk.metadata.get('source', 'unknown')} "
              f"— Page {chunk.metadata.get('page', 'N/A')}")

if __name__ == "__main__":
    question = "What is this document about?"  # ← change this
    run_query(question)
