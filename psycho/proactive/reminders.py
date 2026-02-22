"""
ReminderManager — smart reminders with natural language parsing and proactive delivery.

Reminders are stored in SQLite and checked by the ProactiveScheduler.
They can be created by the user ("remind me to call mom tomorrow at 3pm")
or by the agent ("I'll remind you when your deadline approaches").
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pendulum
from loguru import logger


@dataclass
class Reminder:
    """A single reminder."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    notes: str = ""
    due_timestamp: float = 0.0  # Unix timestamp
    recurrence: str = "none"    # "none" | "daily" | "weekly" | "monthly"
    priority: str = "normal"    # "low" | "normal" | "high" | "urgent"
    completed: bool = False
    snoozed_until: float = 0.0
    created_at: float = field(default_factory=time.time)
    session_id: str = ""

    @property
    def due_dt(self) -> datetime:
        return datetime.fromtimestamp(self.due_timestamp)

    @property
    def is_due(self) -> bool:
        now = time.time()
        if self.completed:
            return False
        if self.snoozed_until and now < self.snoozed_until:
            return False
        return now >= self.due_timestamp

    @property
    def is_upcoming(self) -> bool:
        """Due within the next 2 hours."""
        if self.completed or self.is_due:
            return False
        return (self.due_timestamp - time.time()) <= 7200

    def time_until(self) -> str:
        """Human-readable time until due."""
        delta = self.due_timestamp - time.time()
        if delta < 0:
            return "overdue"
        if delta < 60:
            return "less than a minute"
        if delta < 3600:
            return f"{int(delta/60)} minutes"
        if delta < 86400:
            hours = int(delta / 3600)
            mins = int((delta % 3600) / 60)
            return f"{hours}h {mins}m" if mins else f"{hours} hours"
        days = int(delta / 86400)
        return f"{days} day{'s' if days > 1 else ''}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "notes": self.notes,
            "due_timestamp": self.due_timestamp,
            "recurrence": self.recurrence,
            "priority": self.priority,
            "completed": self.completed,
            "snoozed_until": self.snoozed_until,
            "created_at": self.created_at,
            "session_id": self.session_id,
        }

    @classmethod
    def from_row(cls, row) -> "Reminder":
        return cls(
            id=row["id"],
            title=row["title"],
            notes=row["notes"] or "",
            due_timestamp=row["due_timestamp"],
            recurrence=row["recurrence"] or "none",
            priority=row["priority"] or "normal",
            completed=bool(row["completed"]),
            snoozed_until=row["snoozed_until"] or 0.0,
            created_at=row["created_at"],
            session_id=row["session_id"] or "",
        )


class ReminderManager:
    """
    Manages reminders in SQLite.

    Supports:
    - Natural language time parsing ("tomorrow at 3pm", "in 2 hours")
    - Recurring reminders
    - Snooze
    - Priority levels
    - Due/upcoming queries for proactive delivery
    """

    # Schema migration — called from database connect flow
    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS reminders (
        id              TEXT    PRIMARY KEY,
        title           TEXT    NOT NULL,
        notes           TEXT    DEFAULT '',
        due_timestamp   REAL    NOT NULL,
        recurrence      TEXT    DEFAULT 'none',
        priority        TEXT    DEFAULT 'normal',
        completed       INTEGER DEFAULT 0,
        snoozed_until   REAL    DEFAULT 0,
        created_at      REAL    NOT NULL,
        session_id      TEXT    DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_timestamp, completed);
    """

    def __init__(self, db) -> None:
        self._db = db

    async def ensure_schema(self) -> None:
        await self._db.conn.executescript(self.SCHEMA_SQL)
        await self._db.conn.commit()

    async def create(
        self,
        title: str,
        due_timestamp: float,
        notes: str = "",
        recurrence: str = "none",
        priority: str = "normal",
        session_id: str = "",
    ) -> Reminder:
        r = Reminder(
            title=title,
            notes=notes,
            due_timestamp=due_timestamp,
            recurrence=recurrence,
            priority=priority,
            session_id=session_id,
        )
        await self._db.execute(
            """INSERT INTO reminders
               (id, title, notes, due_timestamp, recurrence, priority, completed,
                snoozed_until, created_at, session_id)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)""",
            (r.id, r.title, r.notes, r.due_timestamp, r.recurrence,
             r.priority, r.created_at, r.session_id),
        )
        logger.info(f"Reminder created: '{r.title}' due {r.due_dt.strftime('%Y-%m-%d %H:%M')}")
        return r

    async def get_due(self) -> list[Reminder]:
        """Return all currently due (or overdue) uncompletd reminders."""
        now = time.time()
        rows = await self._db.fetch_all(
            """SELECT * FROM reminders
               WHERE completed = 0
               AND due_timestamp <= ?
               AND (snoozed_until = 0 OR snoozed_until <= ?)
               ORDER BY priority DESC, due_timestamp ASC""",
            (now, now),
        )
        return [Reminder.from_row(r) for r in rows]

    async def get_upcoming(self, hours_ahead: float = 24.0) -> list[Reminder]:
        """Return reminders due within the next N hours."""
        now = time.time()
        cutoff = now + hours_ahead * 3600
        rows = await self._db.fetch_all(
            """SELECT * FROM reminders
               WHERE completed = 0
               AND due_timestamp BETWEEN ? AND ?
               ORDER BY due_timestamp ASC""",
            (now, cutoff),
        )
        return [Reminder.from_row(r) for r in rows]

    async def get_all_pending(self) -> list[Reminder]:
        rows = await self._db.fetch_all(
            "SELECT * FROM reminders WHERE completed = 0 ORDER BY due_timestamp ASC"
        )
        return [Reminder.from_row(r) for r in rows]

    async def complete(self, reminder_id: str) -> bool:
        cursor = await self._db.execute(
            "UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,)
        )
        return cursor.rowcount > 0

    async def snooze(self, reminder_id: str, minutes: int = 15) -> bool:
        snooze_until = time.time() + minutes * 60
        cursor = await self._db.execute(
            "UPDATE reminders SET snoozed_until = ? WHERE id = ?",
            (snooze_until, reminder_id),
        )
        return cursor.rowcount > 0

    async def reschedule_recurring(self, reminder: Reminder) -> None:
        """If a reminder is recurring, set next occurrence."""
        if reminder.recurrence == "none":
            return
        dt = reminder.due_dt
        if reminder.recurrence == "daily":
            next_dt = dt + timedelta(days=1)
        elif reminder.recurrence == "weekly":
            next_dt = dt + timedelta(weeks=1)
        elif reminder.recurrence == "monthly":
            next_dt = dt.replace(
                month=(dt.month % 12) + 1,
                year=dt.year + (1 if dt.month == 12 else 0),
            )
        else:
            return
        await self._db.execute(
            "UPDATE reminders SET due_timestamp = ?, completed = 0 WHERE id = ?",
            (next_dt.timestamp(), reminder.id),
        )
        logger.info(f"Recurring reminder rescheduled: '{reminder.title}' → {next_dt}")

    async def get_stats(self) -> dict:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) as total, SUM(CASE WHEN completed=0 THEN 1 ELSE 0 END) as pending FROM reminders"
        )
        return {"total": row[0] if row else 0, "pending": row[1] if row else 0}

    def format_for_prompt(self, reminders: list[Reminder]) -> str:
        """Format reminders for injection into system prompt context."""
        if not reminders:
            return ""
        lines = ["─── PENDING REMINDERS ───"]
        priority_icons = {"urgent": "🔴", "high": "🟠", "normal": "🟡", "low": "⚪"}
        for r in reminders[:10]:
            icon = priority_icons.get(r.priority, "⚪")
            time_str = r.due_dt.strftime("%a %b %d at %H:%M")
            overdue = " [OVERDUE]" if r.is_due else f" (in {r.time_until()})"
            lines.append(f"{icon} [{r.priority.upper()}] {r.title} — {time_str}{overdue}")
            if r.notes:
                lines.append(f"   Notes: {r.notes}")
        lines.append("─────────────────────────────")
        return "\n".join(lines)


# ── Natural language time parsing ─────────────────────────────────────────────

_RELATIVE_PATTERNS = [
    # "in X minutes/hours/days"
    (r"in\s+(\d+)\s+minutes?", lambda m: timedelta(minutes=int(m.group(1)))),
    (r"in\s+(\d+)\s+hours?", lambda m: timedelta(hours=int(m.group(1)))),
    (r"in\s+(\d+)\s+days?", lambda m: timedelta(days=int(m.group(1)))),
    (r"in\s+(\d+)\s+weeks?", lambda m: timedelta(weeks=int(m.group(1)))),
    # "in half an hour"
    (r"in\s+half\s+an?\s+hour", lambda m: timedelta(minutes=30)),
    # "in an hour"
    (r"in\s+an?\s+hour", lambda m: timedelta(hours=1)),
]

_ABSOLUTE_PATTERNS = [
    # "tomorrow at HH:MM" or "tomorrow at H am/pm"
    r"tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "today at HH:MM"
    r"today\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "next [weekday] at HH:MM"
    r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "at HH:MM"
    r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
]

_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def parse_reminder_time(text: str) -> Optional[float]:
    """
    Parse a natural-language time expression and return a Unix timestamp.

    Examples:
        "in 30 minutes" → now + 30min
        "tomorrow at 3pm" → tomorrow 15:00
        "next Friday at 9:30am" → next Friday 09:30
        "in 2 hours" → now + 2h
    """
    text = text.lower().strip()
    now = datetime.now()

    # Try relative patterns first
    for pattern, delta_fn in _RELATIVE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            target = now + delta_fn(m)
            return target.timestamp()

    # "tomorrow at ..." / "today at ..."
    m = re.search(r"(tomorrow|today)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if m:
        base = now if m.group(1) == "today" else now + timedelta(days=1)
        hour = int(m.group(2))
        minute = int(m.group(3)) if m.group(3) else 0
        meridiem = m.group(4) or ""
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target.timestamp()

    # "next [weekday] at ..."
    m = re.search(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text,
    )
    if m:
        target_weekday = _WEEKDAY_MAP[m.group(1)]
        hour = int(m.group(2))
        minute = int(m.group(3)) if m.group(3) else 0
        meridiem = m.group(4) or ""
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        days_ahead = (target_weekday - now.weekday() + 7) % 7 or 7
        target_date = now + timedelta(days=days_ahead)
        target = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target.timestamp()

    # "at HH:MM" (assume today if time is in future, else tomorrow)
    m = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        meridiem = m.group(3) or ""
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    return None


def extract_reminder_from_message(message: str) -> Optional[dict]:
    """
    Try to extract a reminder intent from a user message.

    Returns dict with keys: title, due_timestamp, notes, priority
    or None if no reminder detected.
    """
    msg = message.lower()

    # Detect reminder intent
    reminder_triggers = [
        r"remind\s+me\s+to\s+(.+?)(?:\s+(?:at|in|tomorrow|next|on)\b|$)",
        r"remind\s+me\s+(?:at|in|tomorrow|next)\s+",
        r"set\s+(?:a\s+)?reminder\s+(?:to\s+|for\s+)?(.+?)(?:\s+(?:at|in|tomorrow)\b|$)",
        r"don't\s+let\s+me\s+forget\s+(?:to\s+)?(.+?)(?:\s+(?:at|in|tomorrow)\b|$)",
        r"make\s+(?:sure|a\s+note)\s+(?:to\s+|that\s+I\s+)?(.+?)(?:\s+(?:at|in|tomorrow)\b|$)",
    ]

    title = None
    for pattern in reminder_triggers:
        m = re.search(pattern, msg)
        if m and m.lastindex and m.group(1):
            title = m.group(1).strip().rstrip(".,!?")
            break

    if not title and "remind" not in msg and "reminder" not in msg:
        return None

    # If we detected a reminder keyword but couldn't parse the title
    if not title:
        # Try to get it after "remind me to" etc
        for trigger in ["remind me to ", "remind me ", "reminder for ", "reminder to "]:
            if trigger in msg:
                remainder = msg.split(trigger, 1)[1]
                # Trim at time expressions
                for time_word in ["at ", "in ", "tomorrow", "next ", "on "]:
                    if time_word in remainder:
                        idx = remainder.index(time_word)
                        candidate = remainder[:idx].strip()
                        if candidate:
                            title = candidate
                            break
                if not title:
                    title = remainder[:60].strip()
                break

    if not title:
        return None

    # Clean up title
    title = re.sub(r"\s+", " ", title).strip().rstrip(".,!?")
    if len(title) < 2:
        return None

    # Parse time from the full message
    due_timestamp = parse_reminder_time(message)
    if not due_timestamp:
        # Default to 1 hour from now if no time specified
        from datetime import timedelta
        due_timestamp = (datetime.now() + timedelta(hours=1)).timestamp()

    # Detect priority
    priority = "normal"
    if any(w in msg for w in ["urgent", "asap", "immediately", "critical"]):
        priority = "urgent"
    elif any(w in msg for w in ["important", "must", "don't forget", "crucial"]):
        priority = "high"

    return {
        "title": title.capitalize(),
        "due_timestamp": due_timestamp,
        "priority": priority,
        "notes": "",
    }
