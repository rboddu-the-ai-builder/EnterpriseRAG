# retriever.py
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from config import (
    RERANKER_MODEL, TOP_K_RETRIEVAL,
    TOP_K_RERANKED, BM25_WEIGHT, VECTOR_WEIGHT
)

# load reranker once at module level
print(f"Loading reranker model: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL)
print("Reranker ready.")

def build_bm25_index(vectorstore):
    """
    Build a BM25 index from all chunks stored in ChromaDB.
    BM25 works on raw keywords — good for exact term matching.
    """
    # get all documents from ChromaDB
    all_docs = vectorstore.get()
    documents = all_docs["documents"]  # list of chunk texts
    metadatas = all_docs["metadatas"]  # list of metadata dicts

    # BM25 needs tokenized text — split each chunk into words
    tokenized = [doc.lower().split() for doc in documents]

    bm25_index = BM25Okapi(tokenized)

    print(f"BM25 index built with {len(documents)} chunks.")
    return bm25_index, documents, metadatas


def hybrid_search(query, vectorstore, bm25_index, documents, metadatas, k=TOP_K_RETRIEVAL):
    """
    Combines BM25 keyword search + vector semantic search.
    Returns merged results scored by both methods.
    """

    # ── Vector search ────────────────────────────────────────
    vector_results = vectorstore.similarity_search_with_score(query, k=k)
    # returns list of (Document, score) tuples
    # lower score = more similar in ChromaDB

    # ── BM25 keyword search ──────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # returns score for every chunk — higher = more relevant

    # normalize BM25 scores to 0-1 range
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

    # ── Combine both scores ──────────────────────────────────
    # build a dict of chunk_text → combined score
    combined = {}

    # add vector scores
    for doc, score in vector_results:
        # convert distance to similarity (lower distance = higher similarity)
        similarity = 1 - score
        text = doc.page_content
        combined[text] = {
            "doc": doc,
            "score": similarity * VECTOR_WEIGHT
        }

    # add BM25 scores
    for i, (text, metadata) in enumerate(zip(documents, metadatas)):
        bm25_score = bm25_scores_normalized[i] * BM25_WEIGHT
        if text in combined:
            # chunk appeared in both — add scores together
            combined[text]["score"] += bm25_score
        else:
            # chunk only in BM25 results — add it
            from langchain_core.documents import Document
            combined[text] = {
                "doc": Document(page_content=text, metadata=metadata),
                "score": bm25_score
            }

    # sort by combined score, take top k
    sorted_chunks = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:k]

    return [item["doc"] for item in sorted_chunks]


def rerank_chunks(query, chunks, top_k=TOP_K_RERANKED):
    """
    Takes hybrid search results and rescores them using
    a cross-encoder that looks at query + chunk together as a pair.
    Much more accurate than embedding similarity alone.
    """
    if not chunks:
        return []

    # create query-chunk pairs for the cross encoder
    pairs = [(query, chunk.page_content) for chunk in chunks]

    # score each pair
    scores = reranker.predict(pairs)

    # sort by score, keep top_k
    scored_chunks = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    print(f"\nReranking: {len(chunks)} chunks → top {top_k} kept")
    for i, (score, chunk) in enumerate(scored_chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "N/A")
        print(f"  [{i+1}] score={score:.3f} | {source} p.{page}")

    return [chunk for _, chunk in scored_chunks]


def retrieve(query, vectorstore, bm25_index, documents, metadatas):
    """
    Full retrieval pipeline:
    1. Hybrid search (BM25 + vector)
    2. Rerank results
    3. Return top chunks
    """
    # Step 1: hybrid search — gets broader candidate set
    candidates = hybrid_search(
        query, vectorstore, bm25_index, documents, metadatas
    )

    # Step 2: rerank — picks the most relevant from candidates
    final_chunks = rerank_chunks(query, candidates)

    return final_chunks