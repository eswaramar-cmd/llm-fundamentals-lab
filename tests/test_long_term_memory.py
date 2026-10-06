"""Tests for long-term memory operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agent.memory.long_term import LongTermMemory
from backend.agent.state import AgentState


class TestLongTermMemory:
    """Test long-term memory save/retrieve operations."""

    @patch("backend.agent.memory.long_term._get_embeddings")
    @patch("backend.agent.memory.long_term._get_collection")
    def test_save_memory(self, mock_coll, mock_emb):
        mock_vs = MagicMock()
        mock_coll.return_value = mock_vs

        mem = LongTermMemory()
        mem_id = mem.save("user1", "preferred_language", "Python")
        assert "user1" in mem_id
        assert "preferred_language" in mem_id
        mock_vs.add_texts.assert_called_once()

    @patch("backend.agent.memory.long_term._get_embeddings")
    @patch("backend.agent.memory.long_term._get_collection")
    def test_retrieve_memory_empty(self, mock_coll, mock_emb):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_coll.return_value = mock_vs

        mem = LongTermMemory()
        facts = mem.retrieve("user1")
        assert facts == []

    @patch("backend.agent.memory.long_term._get_embeddings")
    @patch("backend.agent.memory.long_term._get_collection")
    def test_retrieve_memory_with_results(self, mock_coll, mock_emb):
        from langchain_core.documents import Document

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(
                page_content="preferred_language: Python",
                metadata={"user_id": "user1", "key": "preferred_language", "value": "Python", "category": "preference"},
            )
        ]
        mock_coll.return_value = mock_vs

        mem = LongTermMemory()
        facts = mem.retrieve("user1")
        assert len(facts) == 1
        assert facts[0]["key"] == "preferred_language"
        assert facts[0]["value"] == "Python"

    @patch("backend.agent.memory.long_term._get_embeddings")
    @patch("backend.agent.memory.long_term._get_collection")
    def test_retrieve_all_memory(self, mock_coll, mock_emb):
        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {
            "metadatas": [
                {"user_id": "user1", "key": "name", "value": "Amar", "category": "identity"},
            ],
            "documents": ["name: Amar"],
        }
        mock_coll.return_value = mock_vs

        mem = LongTermMemory()
        facts = mem.retrieve_all("user1")
        assert len(facts) == 1
        assert facts[0]["key"] == "name"


class TestSaveMemoryTool:
    """Test the save_memory LangChain tool."""

    @patch("backend.agent.memory.long_term._get_embeddings")
    @patch("backend.agent.memory.long_term._get_collection")
    def test_save_memory_tool(self, mock_coll, mock_emb):
        from backend.agent.memory.long_term import save_memory

        mock_vs = MagicMock()
        mock_coll.return_value = mock_vs

        result = save_memory.invoke({
            "user_id": "user1",
            "key": "preferred_language",
            "value": "Python",
            "category": "preference",
        })
        assert "Saved memory" in result
        assert "preferred_language" in result

    @patch("backend.agent.memory.long_term._get_embeddings")
    @patch("backend.agent.memory.long_term._get_collection")
    def test_save_memory_error(self, mock_coll, mock_emb):
        from backend.agent.memory.long_term import save_memory

        # Reset singleton to force re-creation
        import backend.agent.memory.long_term as ltm
        ltm._long_term = None

        mock_coll.side_effect = RuntimeError("Chroma unavailable")
        result = save_memory.invoke({
            "user_id": "user1",
            "key": "test",
            "value": "val",
        })
        assert "error" in result.lower()
