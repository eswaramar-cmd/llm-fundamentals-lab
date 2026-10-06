"""Tests for rule-based task splitting and merging."""

from __future__ import annotations

from backend.agent.parallel import Subtask, merge, split_task


class TestSplitTask:
    def test_single_question_stays_whole(self):
        parts = split_task("What is ChromaDB?")

        assert len(parts) == 1
        assert parts[0].text == "What is ChromaDB?"

    def test_numbered_list(self):
        parts = split_task(
            "1. What is Apex Care? 2. What is ChromaDB? 3. Where is it based?"
        )

        assert len(parts) == 3
        assert [p.text for p in parts] == [
            "What is Apex Care?",
            "What is ChromaDB?",
            "Where is it based?",
        ]

    def test_multiple_questions(self):
        parts = split_task(
            "What is ChromaDB? What does Ollama do? What is the embedding model?"
        )

        assert len(parts) == 3

    def test_then_chain(self):
        parts = split_task(
            "Explain retrieval augmented generation, and then explain embeddings, "
            "and then explain vector search"
        )

        assert len(parts) == 3
        # Order must follow the original request, not length.
        assert parts[0].text.startswith("Explain retrieval")

    def test_no_part_is_lost(self):
        original = "Explain RAG, and then explain embeddings, and then explain vector search"
        parts = split_task(original)

        for fragment in ("RAG", "embeddings", "vector search"):
            assert any(fragment in p.text for p in parts), fragment

    def test_respects_max_subtasks(self):
        from backend.agent.config import get_settings

        limit = get_settings().parallel_max_subtasks
        question = " ".join(f"What is topic number {i}?" for i in range(12))

        parts = split_task(question)

        assert len(parts) <= limit

    def test_arithmetic_is_not_split(self):
        parts = split_task("What is 2 + 2?")

        assert len(parts) == 1

    def test_empty_input(self):
        assert split_task("") == []

    def test_ids_are_sequential(self):
        parts = split_task("1. First question here? 2. Second question here?")

        assert [p.id for p in parts] == list(range(1, len(parts) + 1))

    def test_empty_stays_pending(self):
        parts = split_task("What is ChromaDB?")

        assert parts[0].status == "pending"
        assert parts[0].answer == ""


class TestMerge:
    def test_single_answer_unwrapped(self):
        parts = [Subtask(id=1, text="Q?", status="done", answer="An answer.")]

        assert merge(parts) == "An answer."

    def test_multiple_answers_labelled(self):
        parts = [
            Subtask(id=1, text="First?", status="done", answer="One."),
            Subtask(id=2, text="Second?", status="done", answer="Two."),
        ]

        merged = merge(parts)

        assert "**1. First?**" in merged
        assert "**2. Second?**" in merged
        assert "One." in merged and "Two." in merged

    def test_failed_part_is_reported_not_hidden(self):
        parts = [
            Subtask(id=1, text="First?", status="done", answer="One."),
            Subtask(id=2, text="Second?", status="failed", error="boom"),
        ]

        merged = merge(parts)

        assert "One." in merged
        assert "Incomplete" in merged
        assert "part 2" in merged

    def test_no_answers_gives_empty(self):
        parts = [Subtask(id=1, text="Q?", status="failed", error="boom")]

        assert merge(parts) == ""

    def test_empty_list(self):
        assert merge([]) == ""
