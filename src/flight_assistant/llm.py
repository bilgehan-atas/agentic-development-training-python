"""
Shared local LLM instance, backed by Ollama.

Make sure Ollama is running locally and the model has been pulled:

    ollama pull qwen2.5:7b
    ollama serve

Want to try another model? `OLLAMA_MODEL=llama3.1:8b python -m flight_assistant.main`
(pick one that supports tool calling - see https://ollama.com/search?c=tools)
"""

import os

from langchain_ollama import ChatOllama

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

llm = ChatOllama(model=MODEL, temperature=0)
