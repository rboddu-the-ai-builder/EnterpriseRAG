# Enterprise RAG System

A production-grade Retrieval-Augmented Generation (RAG) system built with LangChain,
ChromaDB, and Ollama. Designed to answer domain-specific questions grounded in source
documents with citations — and explicitly refuse to answer when the documents don't
support a response.

## Overview

This system was built across three phases, progressively adding production-quality
features on top of a working baseline. Drop any PDF documents into the `/docs` folder
and the system will index them, retrieve relevant content, and answer questions with
source citations.

## Features

### Phase 1 — Core Pipeline
- PDF ingestion with automatic chunking (700 tokens, 100 token overlap)
- Semantic embeddings using BAAI/bge-base-en-v1.5 (runs fully locally)
- Vector storage and retrieval with ChromaDB
- Citation-enforced answer generation — every claim references a source
- Explicit refusal when retrieved chunks don't support an answer

### Phase 2 — Production Quality
- Hybrid retrieval combining BM25 keyword search + vector semantic search
- Cross-encoder reranking using ms-marco-MiniLM-L-6-v2
- Versioned prompt templates stored in YAML (prompts/v1.yaml)
- Intelligent ingestion registry — only new files get embedded, no duplicates

### Phase 3 — Evaluation
- Golden evaluation dataset of manually verified Q&A pairs
- RAGAS metrics: faithfulness and answer relevancy
- Quality gates with configurable thresholds

## Project Structure
```
EnterpriseRAG/
├── config.py          # all settings in one place
├── embeddings.py      # embedding model setup
├── ingest.py          # document loading, chunking, ChromaDB storage
├── retriever.py       # hybrid BM25 + vector search + reranking
├── rag.py             # RAG chain with citation enforcement
├── llm.py             # Ollama LLM setup
├── main.py            # run to ingest new documents
├── query.py           # run to ask questions
├── evaluate.py        # run RAGAS evaluation
├── prompts/
│   └── v1.yaml        # versioned prompt template
└── data/
    └── eval_dataset.json  # golden Q&A pairs
```

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Keyword Search | BM25 (rank_bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.2 via Ollama (runs locally) |
| Evaluation | RAGAS |

## Setup

**1 — Clone the repository**
```bash
git clone https://github.com/rakeshboddu/EnterpriseRAG.git
cd EnterpriseRAG
```

**2 — Create and activate virtual environment**
```bash
python -m venv rag-env
rag-env\Scripts\activate        # Windows
source rag-env/bin/activate     # Mac/Linux
```

**3 — Install dependencies**
```bash
pip install langchain langchain-community langchain-huggingface langchain-core \
  langchain-text-splitters chromadb sentence-transformers pypdf \
  python-dotenv rank_bm25 ragas datasets pyyaml
```

**4 — Install Ollama and pull model**

Download Ollama from ollama.com then:
```bash
ollama pull llama3.2
```

**5 — Add your documents**

Place your PDF files in the `/docs` folder.

## Usage

**Ingest documents:**
```bash
python main.py
```
Only new files are embedded — existing files are skipped automatically.

**Ask a question:**
```bash
python query.py
```
Change the `question` variable at the bottom of `query.py`.

**Run evaluation:**
```bash
python evaluate.py
```
Requires a `data/eval_dataset.json` file with manually verified Q&A pairs
in this format:
```json
[
    {
        "question": "Your question here?",
        "ground_truth": "Your verified answer here."
    }
]
```

## How It Works
```
Question
    ↓
Hybrid Search (BM25 + Vector) → top 10 candidate chunks
    ↓
Cross-Encoder Reranker → top 5 most relevant chunks
    ↓
Citation-enforced prompt → Ollama Llama 3.2
    ↓
Answer with source citations [Source: filename, Page: N]
```

Evaluated on 5 manually verified Q&A pairs using RAGAS metrics.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.79 - 0.89 | 0.85 | Varies by run |
| Answer Relevancy | N/A | 0.80 | See note below |

> **Note:** Evaluation uses a local Ollama LLM for answer generation and Groq API 
> for RAGAS scoring. Answer Relevancy could not be consistently calculated as RAGAS 
> internally requests multiple LLM generations simultaneously which Groq's API does 
> not support. Faithfulness scores varied between runs due to occasional timeout 
> errors in RAGAS parallel scoring jobs. A more stable evaluation environment 
> such as OpenAI API would produce consistent scores across all metrics.

*Run `python evaluate.py` against your own document corpus to generate results*

## Key Design Decisions

**Why hybrid retrieval?** Vector search understands meaning but can miss exact
keyword matches — acronyms, proper nouns, specific terms. BM25 catches those.
Together they cover each other's blind spots.

**Why a reranker?** Vector similarity scores chunks independently. A cross-encoder
reads the query and chunk together as a pair, giving a much more accurate relevance
score at the cost of speed. Using it as a second-pass filter gives the best of both.

**Why citation enforcement?** Grounding every claim in retrieved evidence makes the
system trustworthy. If the documents don't support an answer, the system says so
rather than hallucinating.

**Why versioned prompts?** A prompt change can affect system behaviour as
dramatically as a code change. Storing prompts in versioned YAML files means
they are tracked, auditable, and rollback-able — just like code.

## Running Everything Locally

All models run on your machine — no data is sent to external APIs:
- Embeddings: BAAI/bge-base-en-v1.5 downloaded to ~/.cache/huggingface
- Reranker: ms-marco-MiniLM-L-6-v2 downloaded to ~/.cache/huggingface
- LLM: Llama 3.2 via Ollama
- Vector store: ChromaDB persisted to ./chroma_db