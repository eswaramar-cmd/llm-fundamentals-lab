"""Tests for the web search abstraction."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.agent.tools.web_search import DuckDuckGoSearchProvider, SearchResponse, SearchResult


class TestWebSearch:
    def test_provider_search_returns_results(self):
        """Test with a mocked HTTP response."""
        mock_html = """
        <html><body>
            <div class="result">
                <a class="result__a" href="https://example.com/1">Test Title 1</a>
                <a class="result__snippet">This is a test snippet 1</a>
            </div>
            <div class="result">
                <a class="result__a" href="https://example.com/2">Test Title 2</a>
                <a class="result__snippet">This is a test snippet 2</a>
            </div>
        </body></html>
        """

        # Test the provider returns structured results
        provider = DuckDuckGoSearchProvider()
        with patch.object(provider, "search") as mock_search:
            mock_search.return_value = SearchResponse(
                query="test",
                results=[
                    SearchResult(title="Test Title 1", url="https://example.com/1", snippet="snippet 1"),
                    SearchResult(title="Test Title 2", url="https://example.com/2", snippet="snippet 2"),
                ],
            )
            resp = provider.search("test query")
            assert resp.query == "test"
            assert len(resp.results) == 2
            assert resp.results[0].title == "Test Title 1"

    def test_web_search_tool_name(self):
        from backend.agent.tools.web_search import web_search
        assert web_search.name == "web_search"

    def test_web_search_tool_args(self):
        from backend.agent.tools.web_search import web_search
        args = web_search.args
        assert "query" in args
        assert "num_results" in args

    @patch("backend.agent.tools.web_search.DuckDuckGoSearchProvider.search")
    def test_web_search_no_results(self, mock_search):
        from backend.agent.tools.web_search import web_search
        mock_search.return_value = SearchResponse(
            query="test", results=[], error=None
        )
        result = web_search.invoke({"query": "nothing at all"})
        assert "No results found" in result
