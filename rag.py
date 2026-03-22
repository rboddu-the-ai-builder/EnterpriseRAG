# rag.py
import yaml
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import PROMPT_VERSION, PROMPTS_DIR

def load_prompt():
    """Load prompt from versioned yaml file."""
    prompt_file = os.path.join(PROMPTS_DIR, f"{PROMPT_VERSION}.yaml")
    with open(prompt_file, "r") as f:
        prompt_data = yaml.safe_load(f)

    print(f"Loaded prompt version: {prompt_data['version']}")
    return PromptTemplate(
        template=prompt_data["template"],
        input_variables=prompt_data["input_variables"]
    )

def format_chunks(docs):
    """Format retrieved chunks with source info for the prompt."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Chunk {i+1} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

def build_rag_chain(llm):
    """
    Build RAG chain with prompt loaded from yaml.
    Retrieval is now handled separately in retriever.py
    so we build a simpler chain here that just takes
    pre-retrieved chunks + question.
    """
    prompt = load_prompt()

    rag_chain = (
        {
            "context": lambda x: format_chunks(x["chunks"]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain