"""Web search helper with optional SerpAPI and DuckDuckGo fallback."""

from __future__ import annotations
from typing import Any
import requests
from duckduckgo_search import DDGS
from config.config import get_settings


def _search_with_serpapi(query: str, max_results: int) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.search_api_key:
        return []

    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "api_key": settings.search_api_key,
            "num": max_results,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("organic_results", [])


def _search_with_duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def _summarize_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant web results found."

    summary_lines = ["Live web search summary:"]
    for result in results[:5]:
        title = result.get("title", "Untitled")
        snippet = result.get("snippet") or result.get("body") or "No snippet available."
        link = result.get("link") or result.get("href") or ""
        summary_lines.append(f"- {title}: {snippet} {link}".strip())

    return "\n".join(summary_lines)


def search_web(query: str, max_results: int = 5) -> str:
    """Fetch and summarize real-time web search results."""

    try:
        serp_results = _search_with_serpapi(query=query, max_results=max_results)
        if serp_results:
            return _summarize_results(serp_results)

        ddg_results = _search_with_duckduckgo(query=query, max_results=max_results)
        return _summarize_results(ddg_results)
    except Exception as exc:
        return f"Web search unavailable: {exc}"
