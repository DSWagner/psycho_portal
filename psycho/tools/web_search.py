"""Web search — DuckDuckGo (free, no key) + optional Brave API."""

from __future__ import annotations

import re

import httpx
from loguru import logger

# ── Trigger detection ──────────────────────────────────────────────────────────

_SEARCH_PATTERNS = [
    r"\bsearch(?:\s+for)?\s+(.+)",
    r"\blook\s+up\s+(.+)",
    r"\bfind\s+(?:info(?:rmation)?\s+(?:about|on)\s+)?(.+)",
    r"\bwhat[''s]*\s+(?:the\s+)?(?:latest|current|recent)\b",
    r"\bwho\s+is\s+(.+)",
    r"\bwhere\s+is\s+(.+)",
    r"\bprice\s+of\s+(.+)",
    r"\bweather\s+(?:in\s+)?(.+)",
]

_LIVE_KEYWORDS = {
    "today", "current", "latest", "recent", "right now", "breaking",
    "news", "price", "weather", "stock", "trending", "2024", "2025", "2026",
}


def should_search(message: str) -> bool:
    """Return True if the message likely needs live web data."""
    msg_lower = message.lower()
    # Direct search command
    if re.search(r"\b(search|look up|google|bing|find info)\b", msg_lower):
        return True
    # Live-data keywords
    if any(kw in msg_lower for kw in _LIVE_KEYWORDS):
        return True
    return False


def extract_query(message: str) -> str:
    """Pull the best search query from the message."""
    msg_lower = message.lower()
    for pattern in _SEARCH_PATTERNS:
        m = re.search(pattern, msg_lower)
        if m and m.lastindex:
            return m.group(1).strip()[:200]
    return message.strip()[:200]


# ── DuckDuckGo (free) ─────────────────────────────────────────────────────────

async def search_duckduckgo(query: str) -> list[dict]:
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers={"User-Agent": "PsychoPortal/1.0"},
            )
        if r.status_code != 200:
            return results
        data = r.json()

        if abstract := data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Overview"),
                "snippet": abstract[:500],
                "url": data.get("AbstractURL", ""),
                "source": "DuckDuckGo",
            })

        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic["Text"][:80],
                    "snippet": topic["Text"][:350],
                    "url": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo",
                })
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")
    return results[:5]


# ── Brave Search (optional, free tier 2000/month) ────────────────────────────

async def search_brave(query: str, api_key: str) -> list[dict]:
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": "5"},
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            )
        if r.status_code != 200:
            logger.warning(f"Brave Search HTTP {r.status_code}")
            return results
        data = r.json()
        for item in data.get("web", {}).get("results", [])[:5]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", "")[:400],
                "url": item.get("url", ""),
                "source": "Brave",
            })
    except Exception as e:
        logger.warning(f"Brave search error: {e}")
    return results


# ── Unified entry point ───────────────────────────────────────────────────────

async def web_search(query: str, brave_api_key: str = "") -> list[dict]:
    """Run search: Brave if key set, else DuckDuckGo."""
    if brave_api_key:
        results = await search_brave(query, brave_api_key)
        if results:
            return results
    return await search_duckduckgo(query)


def format_search_results(results: list[dict], query: str) -> str:
    """Format search results block for system prompt injection."""
    if not results:
        return ""
    lines = [f"\n─── WEB SEARCH: {query!r} ───"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r['title']}")
        if r.get("snippet"):
            lines.append(f"    {r['snippet']}")
        if r.get("url"):
            lines.append(f"    URL: {r['url']}")
    lines.append("──────────────────────────────────────────────")
    lines.append("Incorporate the above into your answer and cite sources where useful.")
    return "\n".join(lines)
