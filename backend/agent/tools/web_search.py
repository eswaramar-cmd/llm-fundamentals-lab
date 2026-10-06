"""Web-search abstraction — uses DuckDuckGo HTML endpoint (no API key required).

This is intentionally an *abstraction*: the ``WebSearchProvider`` protocol
can be swapped for any search backend (Google, Bing, SerpAPI, etc.) without
touching the agent code.

Falls back gracefully when the web is unavailable — the tool returns a
structured message instead of crashing the agent.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    """Aggregated search response."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    error: str | None = None


class WebSearchProvider(Protocol):
    """Protocol that any web-search backend must implement."""

    def search(self, query: str, num_results: int = 5) -> SearchResponse: ...


class DuckDuckGoSearchProvider:
    """Search via the DuckDuckGo HTML endpoint (no API key needed)."""

    BASE_URL = "https://html.duckduckgo.com/html/"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    def search(self, query: str, num_results: int = 5) -> SearchResponse:
        params = {"q": query, "kl": "us-en"}
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.HEADERS,
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return SearchResponse(
                query=query,
                results=[],
                error=f"Web search unavailable: {type(exc).__name__}",
            )

        return self._parse(resp.text, query, num_results)

    def _parse(self, html: str, query: str, num_results: int) -> SearchResponse:
        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []

        for result in soup.select("div.result"):
            title_tag = result.select_one("a.result__a")
            snippet_tag = result.select_one("a.result__snippet")

            if title_tag:
                title = title_tag.get_text(strip=True)
                url = title_tag.get("href", "")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append(
                    SearchResult(title=title, url=url, snippet=snippet)
                )

            if len(results) >= num_results:
                break

        return SearchResponse(query=query, results=results)


# Module-level provider instance
_provider: WebSearchProvider | None = None


def get_search_provider() -> WebSearchProvider:
    global _provider
    if _provider is None:
        _provider = DuckDuckGoSearchProvider()
    return _provider


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(..., description="Search query", min_length=1, max_length=512)
    num_results: int = Field(default=5, description="Number of results", ge=1, le=10)


def perform_web_search(query: str, num_results: int = 5, provider: WebSearchProvider | None = None) -> str:
    """Execute web search and format results as markdown."""
    if provider is None:
        provider = get_search_provider()

    resp = provider.search(query, num_results)
    if resp.error:
        return f"[web_search error: {resp.error}]"

    if not resp.results:
        return "[web_search: No results found.]"

    parts = [f"Search results for: {resp.query}\n"]
    for i, r in enumerate(resp.results, start=1):
        parts.append(
            f"\n{i}. Title: {r.title}\n"
            f"   URL: {r.url}\n"
            f"   Snippet: {r.snippet}"
        )
    return "\n".join(parts)


def web_search_tool(provider: WebSearchProvider | None = None) -> type:
    """Create a LangChain tool wrapping a WebSearchProvider."""
    return lc_tool(
        "web_search",
        description=(
            "Search the web for current information about a topic. "
            "Use this when the user asks about recent events, current "
            "facts, or anything not in the local knowledge base. "
            "Input: 'query' (search string) and optional 'num_results' (1-10)."
        ),
        args_schema=WebSearchInput,
    )(lambda query, num_results=5: perform_web_search(query, num_results, provider))


# Module-level tool instance
web_search = web_search_tool()

