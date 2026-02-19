"""
Domain router — classifies user intent into one of four domains.
Uses a single cheap LLM call (haiku, ~50 tokens) with caching.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from psycho.llm.base import LLMProvider

DOMAINS = ("coding", "health", "tasks", "general")

ROUTER_PROMPT = """\
Classify this message into exactly one domain.

Message: {message}

Domains:
- coding: programming, software engineering, debugging, algorithms, code review, tech tools
- health: nutrition, diet, fitness, exercise, sleep, weight, medical, wellness, mental health
- tasks: todos, reminders, planning, scheduling, goals, productivity, deadlines
- general: everything else (questions, conversation, knowledge, writing, math)

Return only JSON: {{"domain": "coding|health|tasks|general", "confidence": 0.0-1.0}}"""

# Fast keyword shortcuts — avoid LLM call for obvious cases
_KEYWORD_MAP = {
    "coding": {
        "python", "javascript", "typescript", "java", "c++", "rust", "go", "code",
        "function", "class", "bug", "error", "exception", "debug", "git", "github",
        "api", "database", "sql", "async", "await", "import", "library", "framework",
        "deploy", "docker", "kubernetes", "algorithm", "refactor", "test", "unittest",
        "def ", "const ", "var ", "let ", "fn ", "func ", "import ", "from ",
    },
    "health": {
        "weight", "kg", "lbs", "calories", "calorie", "sleep", "slept", "exercise",
        "workout", "gym", "run", "walk", "diet", "nutrition", "meal", "eat", "ate",
        "drink", "water", "steps", "bmi", "heart rate", "mood", "stress", "anxiety",
        "tired", "fatigue", "protein", "carb", "fat", "vitamin", "supplement",
        "health", "medical", "doctor", "pain", "sick",
    },
    "tasks": {
        "remind", "reminder", "todo", "to do", "to-do", "task", "plan", "schedule",
        "deadline", "due", "finish", "complete", "done", "priority", "urgent",
        "tomorrow", "next week", "don't forget", "need to", "have to", "should",
        "appointment", "meeting", "call", "follow up",
    },
}


class DomainRouter:
    """
    Routes user messages to the appropriate domain handler.

    Strategy (in order):
        1. Keyword matching (fast, no LLM call) — covers ~70% of cases
        2. LLM classification (haiku, ~50 tokens) — for ambiguous cases
        3. Fallback to "general"
    """

    def __init__(self, llm: "LLMProvider") -> None:
        self._llm = llm
        self._cache: dict[str, str] = {}  # message hash → domain

    async def classify(self, user_message: str) -> str:
        """Classify user message into a domain. Returns domain string."""
        # Cache check (first 100 chars as key)
        cache_key = user_message[:100].lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        domain = self._keyword_classify(user_message)

        if domain == "general" and len(user_message) > 15:
            # Try LLM for non-obvious cases
            domain = await self._llm_classify(user_message)

        self._cache[cache_key] = domain
        logger.debug(f"Domain: '{domain}' for '{user_message[:50]}'")
        return domain

    def _keyword_classify(self, message: str) -> str:
        """Fast keyword-based classification."""
        msg_lower = message.lower()
        scores = {domain: 0 for domain in DOMAINS}

        for domain, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                if kw in msg_lower:
                    scores[domain] += 1

        best_domain = max(scores, key=lambda d: scores[d])
        if scores[best_domain] >= 2:
            return best_domain
        if scores[best_domain] == 1:
            return best_domain  # single keyword match is good enough

        return "general"

    async def _llm_classify(self, user_message: str) -> str:
        """LLM-based classification for ambiguous messages."""
        from psycho.llm.base import Message
        try:
            response = await self._llm.complete(
                messages=[
                    Message(
                        role="user",
                        content=ROUTER_PROMPT.format(message=user_message[:300]),
                    )
                ],
                system="Output ONLY valid JSON. No explanation.",
                max_tokens=50,
                temperature=0.0,
            )
            raw = response.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(raw)
            domain = data.get("domain", "general")
            return domain if domain in DOMAINS else "general"
        except Exception as e:
            logger.debug(f"Domain LLM classify failed (using general): {e}")
            return "general"
