"""AI Research & Knowledge Agent.

A production-oriented, LangGraph-based AI agent that combines:
- ChromaDB RAG (knowledge base search over PDF documents)
- Ollama LLM (llama3.2:3b)
- Tool calling (calculator, knowledge base, web search)
- Short-term session memory (LangGraph checkpoint memory)
- Long-term memory (ChromaDB-based persistent facts)
- Human-in-the-loop (LangGraph interrupt/resume)
- Streaming, retry, and error handling
"""
