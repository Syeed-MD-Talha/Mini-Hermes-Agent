"""
Free web search and fetch tools.

Uses duckduckgo-search (no API key required) for search, and
requests + BeautifulSoup for fetching page content.
"""

from __future__ import annotations

import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


DEFAULT_TIMEOUT = 30
MAX_FETCH_CHARS = 80_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class WebTools:
    """No-API-key web search and page fetch primitives."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def web_search(self, query: str, num_results: int = 5) -> dict:
        """Search the web via DuckDuckGo (no API key required)."""
        try:
            with DDGS() as ddgs:
                raw_results = ddgs.text(query, max_results=min(num_results, 10))
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in raw_results
                ]
            return {
                "ok": True,
                "query": query,
                "results": results,
            }
        except Exception as e:
            return {"ok": False, "error": f"web_search failed: {e}"}

    def web_fetch(self, url: str) -> dict:
        """Fetch and extract readable text from a URL."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                text = response.text
            else:
                text = self._extract_text(response.text)

            if len(text) > MAX_FETCH_CHARS:
                head = text[: MAX_FETCH_CHARS // 2]
                tail = text[-(MAX_FETCH_CHARS // 2 - 200) :]
                text = f"{head}\n... [truncated] ...\n{tail}"

            return {
                "ok": True,
                "url": url,
                "title": self._extract_title(response.text),
                "content": text,
            }
        except Exception as e:
            return {"ok": False, "error": f"web_fetch failed: {e}"}

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements.
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Prefer article/main content if available.
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            main = soup

        text = main.get_text(separator="\n")
        # Collapse blank lines.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def _extract_title(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        return WebTools._clean_text(title_tag.get_text()) if title_tag else ""

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def schemas(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web using DuckDuckGo Lite (no API key required). "
                        "Returns a list of result titles, URLs, and snippets. "
                        "Use this for current events, weather, news, or facts that may change over time."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query.",
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default 5, max 10).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": (
                        "Fetch and extract readable text from a specific URL. "
                        "Use after web_search to read a page in detail."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to fetch.",
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict) -> dict:
        if name == "web_search":
            return self.web_search(
                arguments.get("query", ""),
                min(int(arguments.get("num_results", 5)), 10),
            )
        if name == "web_fetch":
            return self.web_fetch(arguments.get("url", ""))
        return {"ok": False, "error": f"unknown web tool: {name}"}
