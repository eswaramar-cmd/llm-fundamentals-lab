"""Tests for keyword-based classification (fast path)."""

from __future__ import annotations

from backend.agent.nodes.router import _keyword_classify


class TestKeywordClassification:
    """Test the fast keyword-based classifier."""

    def test_math_calculation(self):
        """Math expressions should classify as 'calculate'."""
        assert _keyword_classify("What is 25 * 50?") == "calculate"
        assert _keyword_classify("Calculate 2 + 2") == "calculate"
        assert _keyword_classify("What is 100 plus 30?") == "calculate"
        assert _keyword_classify("Multiply 7 by 8") == "calculate"
        assert _keyword_classify("25 % 3") == "calculate"

    def test_rag_query(self):
        """Questions about documents should classify as 'rag'."""
        assert _keyword_classify("What is written in my PDF?") == "rag"
        assert _keyword_classify("Tell me about Apex Care from my documents") == "rag"
        assert _keyword_classify("Search my knowledge base for ChromaDB") == "rag"

    def test_research_query(self):
        """Current events queries should classify as 'research'."""
        assert _keyword_classify("What is the latest AI news?") == "research"
        assert _keyword_classify("Current state of LLMs in 2025") == "research"

    def test_direct_greeting(self):
        """Greetings should classify as 'direct'."""
        assert _keyword_classify("Hello") == "direct"
        assert _keyword_classify("Hi there") == "direct"
        assert _keyword_classify("How are you?") == "direct"
        assert _keyword_classify("Good morning") == "direct"

    def test_unknown_query(self):
        """Unclassifiable queries return None (fallback to LLM)."""
        result = _keyword_classify("Tell me about quantum computing")
        assert result is None
