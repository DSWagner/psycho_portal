"""
Health domain handler — auto-logs health metrics from conversations,
tracks trends, and provides structured health summaries.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from loguru import logger

from .base import DomainHandler, DomainResult

if TYPE_CHECKING:
    from psycho.agent.context import AgentContext
    from psycho.llm.base import LLMProvider
    from psycho.storage.database import Database


# ── Metric extraction patterns ────────────────────────────────────────────────

@dataclass
class MetricMatch:
    metric_type: str
    value: float
    unit: str
    raw: str


_METRIC_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Weight
    ("weight", re.compile(r"(\d+(?:\.\d+)?)\s*(kg|lbs?|pounds?|kilograms?)", re.I), "kg"),
    # Sleep
    ("sleep", re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s*)?sleep", re.I), "hours"),
    ("sleep", re.compile(r"slept\s*(?:for\s*)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", re.I), "hours"),
    # Calories
    ("calories", re.compile(r"(\d+(?:\.\d+)?)\s*(?:kcal|calories?|cals?)", re.I), "kcal"),
    ("calories", re.compile(r"(?:ate|eaten|consumed)\s*(?:about\s*)?(\d+)\s*(?:kcal|calories?)", re.I), "kcal"),
    # Steps
    ("steps", re.compile(r"(\d[\d,]*)\s*steps?", re.I), "steps"),
    ("steps", re.compile(r"walked\s*(?:about\s*)?(\d[\d,]*)\s*steps?", re.I), "steps"),
    # Water
    ("water", re.compile(r"(\d+(?:\.\d+)?)\s*(?:liters?|litres?|l)\s*(?:of\s*)?water", re.I), "liters"),
    ("water", re.compile(r"drank\s*(\d+)\s*(?:glasses?|cups?)\s*(?:of\s*)?water", re.I), "glasses"),
    # Heart rate
    ("heart_rate", re.compile(r"heart\s*rate\s*(?:of\s*|was\s*|:?\s*)(\d+)", re.I), "bpm"),
    ("heart_rate", re.compile(r"(\d+)\s*bpm", re.I), "bpm"),
    # Mood (1-10)
    ("mood", re.compile(r"mood\s*(?:is\s*|was\s*|:?\s*)(\d+)(?:\s*/\s*10)?", re.I), "/10"),
    ("mood", re.compile(r"feeling\s*(?:like\s*)?(?:a\s*)?(\d+)(?:\s*/\s*10)", re.I), "/10"),
    # Body fat
    ("body_fat", re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:body\s*)?fat", re.I), "%"),
    # Exercise duration
    ("exercise_min", re.compile(r"(\d+)\s*(?:minutes?|mins?)\s*(?:of\s*)?(?:exercise|workout|training|running|cycling)", re.I), "minutes"),
]

HEALTH_SYSTEM_ADDENDUM = """\
For health questions:
- Reference any logged metrics naturally ("Based on your logged weight of X...")
- Be encouraging but realistic — never shame body weight or food choices
- Always recommend consulting a professional for medical decisions
- Use metric units by default (kg, km, liters)"""


class HealthTracker:
    """Logs and retrieves health metrics from SQLite."""

    def __init__(self, db: "Database") -> None:
        self._db = db

    async def log_metric(
        self,
        metric_type: str,
        value: float,
        unit: str,
        notes: str = "",
        session_id: str = "",
    ) -> str:
        metric_id = str(uuid.uuid4())
        await self._db.execute(
            """INSERT INTO health_metrics (id, metric_type, value, unit, notes, timestamp, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (metric_id, metric_type, value, unit, notes, time.time(), session_id),
        )
        logger.info(f"Health metric logged: {metric_type}={value}{unit}")
        return metric_id

    async def get_recent(
        self, metric_type: str, days: int = 30, limit: int = 30
    ) -> list[dict]:
        since = time.time() - (days * 86400)
        rows = await self._db.fetch_all(
            """SELECT * FROM health_metrics
               WHERE metric_type=? AND timestamp >= ?
               ORDER BY timestamp DESC LIMIT ?""",
            (metric_type, since, limit),
        )
        return [dict(r) for r in rows]

    async def get_summary(self, days: int = 7) -> dict:
        """Build a summary of recent metrics for context injection."""
        since = time.time() - (days * 86400)
        rows = await self._db.fetch_all(
            """SELECT metric_type, AVG(value) as avg_val, MAX(value) as max_val,
                      MIN(value) as min_val, COUNT(*) as count, unit
               FROM health_metrics WHERE timestamp >= ?
               GROUP BY metric_type""",
            (since,),
        )
        summary = {}
        for r in rows:
            summary[r["metric_type"]] = {
                "avg": round(r["avg_val"], 2),
                "max": round(r["max_val"], 2),
                "min": round(r["min_val"], 2),
                "count": r["count"],
                "unit": r["unit"],
            }
        return summary

    async def get_latest(self, metric_type: str) -> dict | None:
        row = await self._db.fetch_one(
            "SELECT * FROM health_metrics WHERE metric_type=? ORDER BY timestamp DESC LIMIT 1",
            (metric_type,),
        )
        return dict(row) if row else None

    async def get_stats(self) -> dict:
        row = await self._db.fetch_one("SELECT COUNT(*), COUNT(DISTINCT metric_type) FROM health_metrics")
        if row:
            return {"total_entries": row[0], "metric_types": row[1]}
        return {"total_entries": 0, "metric_types": 0}


def extract_metrics(text: str) -> list[MetricMatch]:
    """Extract all health metrics from text using pattern matching."""
    found = []
    text_lower = text.lower()
    for metric_type, pattern, default_unit in _METRIC_PATTERNS:
        for match in pattern.finditer(text):
            try:
                raw_value = match.group(1).replace(",", "")
                value = float(raw_value)
                # Try to get the unit from the match
                full_match = match.group(0)
                found.append(MetricMatch(
                    metric_type=metric_type,
                    value=value,
                    unit=default_unit,
                    raw=full_match,
                ))
            except (ValueError, IndexError):
                continue
    # Deduplicate by metric_type (keep first match)
    seen = set()
    unique = []
    for m in found:
        if m.metric_type not in seen:
            seen.add(m.metric_type)
            unique.append(m)
    return unique


class HealthHandler(DomainHandler):
    """
    Domain handler for health-related interactions.

    Features:
        - Auto-detects and logs health metrics from user messages
        - Injects recent health context into system prompt
        - Builds trend summaries for agent context
        - Rich-formatted metric display after logging
    """

    def __init__(self, db: "Database", llm: "LLMProvider") -> None:
        super().__init__(db, llm)
        self._tracker = HealthTracker(db)

    @property
    def domain_name(self) -> str:
        return "health"

    def system_addendum(self, ctx: "AgentContext") -> str:
        return HEALTH_SYSTEM_ADDENDUM

    async def get_context_for_prompt(self, session_id: str) -> str:
        """Get recent health metrics for system prompt injection."""
        summary = await self._tracker.get_summary(days=30)
        if not summary:
            return ""
        lines = ["\n─── HEALTH METRICS (last 30 days) ───"]
        for metric, stats in summary.items():
            lines.append(
                f"  {metric}: avg={stats['avg']}{stats['unit']} "
                f"(min={stats['min']}, max={stats['max']}, n={stats['count']})"
            )
        lines.append("─────────────────────────────────────")
        return "\n".join(lines)

    async def post_process(
        self, ctx: "AgentContext", response: str
    ) -> DomainResult:
        result = DomainResult(domain="health")

        # Extract metrics from USER message (what they're reporting)
        metrics = extract_metrics(ctx.user_message)

        logged_metrics = []
        for metric in metrics:
            await self._tracker.log_metric(
                metric_type=metric.metric_type,
                value=metric.value,
                unit=metric.unit,
                notes=ctx.user_message[:200],
                session_id=ctx.session_id,
            )
            logged_metrics.append(metric)
            result.add_action(f"Logged {metric.metric_type}: {metric.value}{metric.unit}")

        if logged_metrics:
            # Build a compact display of logged metrics
            metric_strs = [
                f"{m.metric_type}: {m.value}{m.unit}" for m in logged_metrics
            ]
            result.add_extra(
                f"[green dim]  Logged:[/green dim] {', '.join(metric_strs)}"
            )

        result.structured_data["logged_metrics"] = [
            {"type": m.metric_type, "value": m.value, "unit": m.unit}
            for m in logged_metrics
        ]

        return result
