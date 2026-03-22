# evaluate.py
import json
import os
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


# tell RAGAS to use Ollama instead of OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(Ollama(model="llama3.2"))
ragas_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="llama3.2"))

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
        embeddings=ragas_embeddings
    )

    # Step 5: print results and check against thresholds
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:     {results['faithfulness']:.2f}  (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"Answer Relevancy: {results['answer_relevancy']:.2f}  (threshold: {RELEVANCY_THRESHOLD})")
    print("=" * 50)

    # Step 6: check if we passed quality gates
    passed = True
    if results["faithfulness"] < FAITHFULNESS_THRESHOLD:
        print(f"❌ FAILED: Faithfulness below threshold")
        passed = False
    if results["answer_relevancy"] < RELEVANCY_THRESHOLD:
        print(f"❌ FAILED: Answer relevancy below threshold")
        passed = False

    if passed:
        print("✓ PASSED: All metrics above threshold")

    # Step 7: save results to file so you can track over time
    output = {
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "passed": passed,
        "num_questions": len(eval_data)
    }
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()