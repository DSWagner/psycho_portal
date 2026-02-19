"""
Tasks domain handler — natural language task creation, management, and reminders.
The agent automatically creates tasks from user messages like "remind me to X".
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from .base import DomainHandler, DomainResult

if TYPE_CHECKING:
    from psycho.agent.context import AgentContext
    from psycho.llm.base import LLMProvider
    from psycho.storage.database import Database

# ── Task creation detection ───────────────────────────────────────────────────

_TASK_CREATION_PATTERNS = re.compile(
    r"\b("
    r"remind me to|don'?t let me forget|remember to|"
    r"i need to|i have to|i should|i must|"
    r"need to remember|TODO:|to-?do:|task:|"
    r"add (?:a )?(?:task|reminder|todo) (?:to|for)|"
    r"make (?:a )?note (?:to|that)|"
    r"schedule (?:a|to)|"
    r"follow up (?:on|with|about)"
    r")\b",
    re.IGNORECASE,
)

_PRIORITY_PATTERNS = {
    "urgent": re.compile(r"\b(urgent|asap|immediately|right away|critical|emergency)\b", re.I),
    "high": re.compile(r"\b(important|high priority|priorit(?:y|ize)|soon)\b", re.I),
    "low": re.compile(r"\b(whenever|eventually|low priority|not urgent|someday)\b", re.I),
}

_DUE_PATTERNS = {
    "today": re.compile(r"\b(today|tonight|this evening|before end of day)\b", re.I),
    "tomorrow": re.compile(r"\b(tomorrow|next morning)\b", re.I),
    "this_week": re.compile(r"\b(this week|by friday|by end of week)\b", re.I),
    "next_week": re.compile(r"\b(next week|next monday)\b", re.I),
}

TASKS_SYSTEM_ADDENDUM = """\
For task/planning questions:
- When the user mentions something they need to do, proactively offer to add it as a task
- Reference open tasks when planning daily priorities
- Be specific about deadlines and priorities"""


class TaskManager:
    """Creates and manages tasks in SQLite."""

    def __init__(self, db: "Database") -> None:
        self._db = db

    async def create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "normal",
        due_date: str | None = None,
        tags: list[str] | None = None,
        session_id: str = "",
    ) -> str:
        task_id = str(uuid.uuid4())
        now = time.time()
        await self._db.execute(
            """INSERT INTO tasks
               (id, title, description, priority, status, due_date, tags, created_at, updated_at, session_id)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
            (
                task_id, title[:300], description[:500],
                priority, due_date, json.dumps(tags or []),
                now, now, session_id,
            ),
        )
        logger.info(f"Task created: '{title[:50]}' [{priority}]")
        return task_id

    async def complete_task(self, task_id: str) -> bool:
        now = time.time()
        await self._db.execute(
            "UPDATE tasks SET status='completed', completed_at=?, updated_at=? WHERE id=?",
            (now, now, task_id),
        )
        return True

    async def get_pending(
        self, priority: str | None = None, limit: int = 20
    ) -> list[dict]:
        if priority:
            rows = await self._db.fetch_all(
                "SELECT * FROM tasks WHERE status='pending' AND priority=? ORDER BY created_at ASC LIMIT ?",
                (priority, limit),
            )
        else:
            # Order: urgent > high > normal > low
            rows = await self._db.fetch_all(
                """SELECT * FROM tasks WHERE status='pending'
                   ORDER BY CASE priority
                     WHEN 'urgent' THEN 0 WHEN 'high' THEN 1
                     WHEN 'normal' THEN 2 ELSE 3 END,
                   created_at ASC LIMIT ?""",
                (limit,),
            )
        return [dict(r) for r in rows]

    async def get_all(self, status: str = "pending", limit: int = 50) -> list[dict]:
        rows = await self._db.fetch_all(
            "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        )
        return [dict(r) for r in rows]

    async def get_pending_summary(self, max_items: int = 5) -> str:
        """Short summary for system prompt injection."""
        pending = await self.get_pending(limit=max_items)
        if not pending:
            return ""
        lines = [f"\n─── PENDING TASKS ({len(pending)}) ───"]
        for t in pending:
            due = f" [due: {t['due_date']}]" if t.get("due_date") else ""
            lines.append(f"  [{t['priority'].upper()}] {t['title'][:60]}{due}")
        lines.append("─────────────────────────────────────")
        return "\n".join(lines)

    async def get_stats(self) -> dict:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) FROM tasks WHERE status='pending'"
        )
        pending = row[0] if row else 0
        row2 = await self._db.fetch_one("SELECT COUNT(*) FROM tasks")
        total = row2[0] if row2 else 0
        return {"pending_tasks": pending, "total_tasks": total}

    async def search_by_title(self, query: str) -> list[dict]:
        rows = await self._db.fetch_all(
            "SELECT * FROM tasks WHERE title LIKE ? AND status='pending' LIMIT 5",
            (f"%{query}%",),
        )
        return [dict(r) for r in rows]


def _extract_task_title(user_message: str) -> str:
    """Extract the task title from a message like 'remind me to call John'."""
    # Remove the trigger phrase
    cleaned = _TASK_CREATION_PATTERNS.sub("", user_message).strip()
    # Clean up leading connectives
    cleaned = re.sub(r"^(to |that |about |for |a |the )", "", cleaned, flags=re.I).strip()
    # Capitalize
    return cleaned[:200].capitalize() if cleaned else user_message[:100]


def _detect_priority(text: str) -> str:
    for priority, pattern in _PRIORITY_PATTERNS.items():
        if pattern.search(text):
            return priority
    return "normal"


def _detect_due_date(text: str) -> str | None:
    from datetime import date
    today = date.today()
    if _DUE_PATTERNS["today"].search(text):
        return today.isoformat()
    if _DUE_PATTERNS["tomorrow"].search(text):
        return (today + timedelta(days=1)).isoformat()
    if _DUE_PATTERNS["this_week"].search(text):
        days_until_friday = (4 - today.weekday()) % 7 or 7
        return (today + timedelta(days=days_until_friday)).isoformat()
    if _DUE_PATTERNS["next_week"].search(text):
        days_until_monday = (7 - today.weekday()) % 7 or 7
        return (today + timedelta(days=days_until_monday)).isoformat()
    return None


from datetime import timedelta


class TaskHandler(DomainHandler):
    """
    Domain handler for task/planning interactions.

    Features:
        - Auto-detects task creation requests in user messages
        - Creates tasks automatically with inferred priority and due date
        - Injects pending tasks into system prompt
        - Provides task management commands
    """

    def __init__(self, db: "Database", llm: "LLMProvider") -> None:
        super().__init__(db, llm)
        self.manager = TaskManager(db)

    @property
    def domain_name(self) -> str:
        return "tasks"

    def system_addendum(self, ctx: "AgentContext") -> str:
        return TASKS_SYSTEM_ADDENDUM

    async def get_context_for_prompt(self, session_id: str) -> str:
        """Get pending tasks for system prompt injection."""
        return await self.manager.get_pending_summary(max_items=5)

    async def post_process(
        self, ctx: "AgentContext", response: str
    ) -> DomainResult:
        result = DomainResult(domain="tasks")

        # Check if user is creating a task
        if _TASK_CREATION_PATTERNS.search(ctx.user_message):
            title = _extract_task_title(ctx.user_message)
            priority = _detect_priority(ctx.user_message)
            due_date = _detect_due_date(ctx.user_message)

            if len(title) > 5:  # sanity check
                task_id = await self.manager.create_task(
                    title=title,
                    priority=priority,
                    due_date=due_date,
                    session_id=ctx.session_id,
                )
                result.structured_data["task_created"] = {
                    "id": task_id,
                    "title": title,
                    "priority": priority,
                    "due_date": due_date,
                }
                due_str = f" [due {due_date}]" if due_date else ""
                result.add_action(
                    f"Task created: '{title[:50]}' [{priority}]{due_str}"
                )
                result.add_extra(
                    f"[green dim]  Task added:[/green dim] {title[:60]} [{priority}]{due_str}"
                )

        return result
