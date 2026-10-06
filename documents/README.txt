Apex Care — AI Research & Knowledge Agent Documentation

Overview
========
Apex Care is a technology company focused on AI research and knowledge management.
The company provides tools for researchers and developers to build intelligent agents.

Technology Stack
================
- LLM: llama3.2:3b (running via Ollama)
- Embeddings: nomic-embed-text (Ollama)
- Vector Database: ChromaDB
- Orchestration: LangGraph
- API Framework: FastAPI
- Frontend: Vanilla JavaScript + HTML

Architecture
============
The system is organized as follows:

1. FastAPI receives HTTP requests at /chat and /agent/run endpoints.
2. LangGraph orchestrates the agent with memory, RAG, and tool calling.
3. ChromaDB stores document embeddings for knowledge base search.
4. Ollama provides both the LLM (llama3.2:3b) and embeddings (nomic-embed-text).
5. The agent uses tools: calculator, knowledge_base_search, web_search, save_memory.
6. Human-in-the-loop uses LangGraph interrupt/resume for sensitive operations.

Memory
======
Short-term memory uses LangGraph's MemorySaver (in-memory checkpoint).
Long-term memory uses ChromaDB to store persistent user facts and preferences.

Example:
  User: "My name is Amar."
  Agent: "Nice to meet you, Amar."
  User: "What is my name?"
  Agent: "Your name is Amar."

Tools
=====
- calculator: Evaluates arithmetic expressions safely (no code execution).
- knowledge_base_search: Searches ChromaDB for relevant documents.
- web_search: Searches the web via DuckDuckGo (no API key needed).
- save_memory: Stores persistent user facts in ChromaDB.
