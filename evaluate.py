# evaluate.py
import json
import os
import numpy as np

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from embeddings import load_embedding_model
from ingest import load_or_create_vectorstore
from llm import load_llm
from rag import build_rag_chain, format_chunks
from retriever import build_bm25_index, retrieve
from config import EVAL_DATASET_FILE, FAITHFULNESS_THRESHOLD, RELEVANCY_THRESHOLD
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

from dotenv import load_dotenv
load_dotenv()  # loads .env file

# tell RAGAS to use Ollama instead of OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant"))
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
)


def run_evaluation():
    # Step 1: load everything
    print("Loading models and vector store...")
    embeddings = load_embedding_model()
    vectorstore = load_or_create_vectorstore(embeddings)

    if vectorstore is None:
        print("No vector store found. Run main.py first.")
        return

    llm = load_llm()
    rag_chain = build_rag_chain(llm)

    bm25_index, documents, metadatas = build_bm25_index(vectorstore)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Step 2: load your golden dataset
    print("Loading evaluation dataset...")
    with open(EVAL_DATASET_FILE, "r") as f:
        eval_data = json.load(f)

    eval_data = eval_data[:5]  # ← including only 5 questions from the eval dataset

    print(f"Running evaluation on {len(eval_data)} questions...")

    # Step 3: run each question through RAG and collect results
    questions = []
    answers = []
    contexts = []
    ground_truths = []

# Change this
    for i, item in enumerate(eval_data):
        question = item["question"]
        print(f"  [{i+1}/{len(eval_data)}] {question}")

        chunks = retrieve(
            question,
            vectorstore,
            bm25_index,
            documents,
            metadatas
        )

        if not chunks:
            print(f"  → No chunks retrieved, skipping.")
            continue

        # get answer from RAG chain
        answer = rag_chain.invoke({
            "chunks": chunks,
            "question": question
        })

        chunk_texts = [c.page_content for c in chunks]

        questions.append(question)
        answers.append(answer)
        contexts.append(chunk_texts)
        ground_truths.append(item["ground_truth"])
    # Step 4: run RAGAS evaluation
    print("\nRunning RAGAS metrics...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=120,
        )
    )

    # Step 5: print results and check against thresholds

    faithfulness_score = float(np.nanmean(results['faithfulness']))
    relevancy_score = float(np.nanmean(results['answer_relevancy']))
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:     {faithfulness_score:.2f}  (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"Answer Relevancy: {relevancy_score:.2f}  (threshold: {RELEVANCY_THRESHOLD})")
    print("=" * 50)

    # Step 6: check if we passed quality gates
    # Quality gates

    passed = True

    if faithfulness_score < FAITHFULNESS_THRESHOLD:
        print(f"❌ FAILED: Faithfulness {faithfulness_score:.2f} below threshold {FAITHFULNESS_THRESHOLD}")
        passed = False
    if np.isnan(relevancy_score):
        print(f"⚠️  Answer Relevancy: could not be calculated due to timeouts")
    else:
        if relevancy_score < RELEVANCY_THRESHOLD:
            print(f"❌ FAILED: Relevancy {relevancy_score:.2f} below threshold {RELEVANCY_THRESHOLD}")
            passed = False

    if passed:
        print("✓ PASSED: All metrics above threshold")

    # Step 7: save results to file so you can track over time
    output = {
        "faithfulness": faithfulness_score,
        "answer_relevancy": None if np.isnan(relevancy_score) else relevancy_score,
        "passed": passed,
        "num_questions": len(questions)
    }
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()