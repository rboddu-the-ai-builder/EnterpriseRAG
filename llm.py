# llm.py
from langchain_community.llms import Ollama

def load_llm():
    print("Loading Ollama LLM...")
    llm = Ollama(model="llama3.2")
    print("LLM ready.")
    return llm