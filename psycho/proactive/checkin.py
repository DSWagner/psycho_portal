"""
CheckinEngine — context-aware proactive check-ins.

The agent should feel like a person who genuinely cares and checks in.
Check-ins are triggered based on:
- Time of day patterns
- User's stress signals from recent interactions
- Long gaps between sessions
- After completing important tasks or reminders
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger


@dataclass
class CheckinRecord:
    """A record of a proactive check-in that was delivered."""
    timestamp: float = field(default_factory=time.time)
    checkin_type: str = ""  # "morning", "evening", "gap", "stress", "task_done"
    message: str = ""
    acknowledged: bool = False


class CheckinEngine:
    """
    Generates contextual check-in messages when the user returns after a gap,
    at certain times of day, or after detecting stress/difficulty.

    Check-ins are injected as the first message in a new session or as
    proactive insertions during an active session.
    """

    def __init__(self) -> None:
        self._last_checkin: float = 0.0
        self._last_session_end: float = 0.0
        self._daily_checkin_sent: dict[str, bool] = {}  # date → sent
        self._stress_count: int = 0  # Consecutive stress signals detected
        self._checkin_history: list[CheckinRecord] = []

    def record_session_end(self) -> None:
        self._last_session_end = time.time()

    def record_stress(self) -> None:
        self._stress_count += 1

    def reset_stress(self) -> None:
        self._stress_count = 0

    def record_checkin_sent(self, checkin_type: str, message: str) -> None:
        self._last_checkin = time.time()
        self._checkin_history.append(
            CheckinRecord(checkin_type=checkin_type, message=message)
        )
        date_key = datetime.now().strftime("%Y-%m-%d")
        self._daily_checkin_sent[date_key] = True

    def should_checkin(self, session_gap_hours: Optional[float] = None) -> Optional[str]:
        """
        Determine if a check-in is warranted. Returns check-in type or None.

        Check-in types:
        - "morning_greeting": First interaction of the day in morning hours
        - "evening_checkin": Evening session after a busy day
        - "long_gap": Returning after days of absence
        - "stress_followup": User seemed stressed in last session
        """
        now = datetime.now()
        date_key = now.strftime("%Y-%m-%d")

        # Don't send multiple check-ins per day (except stress followup)
        already_sent_today = self._daily_checkin_sent.get(date_key, False)

        # Long gap — returning after 48+ hours
        if session_gap_hours and session_gap_hours >= 48:
            return "long_gap"

        # Morning greeting (first check-in of the day, 6am-11am)
        if not already_sent_today and 6 <= now.hour < 11:
            return "morning_greeting"

        # Evening check-in (after 6pm, first check-in of evening)
        if not already_sent_today and 18 <= now.hour < 23:
            return "evening_checkin"

        # Stress followup (if last session had stress signals)
        if self._stress_count >= 2 and not already_sent_today:
            return "stress_followup"

        return None

    def generate_checkin_context(
        self,
        checkin_type: str,
        user_name: str = "",
        last_projects: Optional[list[str]] = None,
        pending_reminders: int = 0,
        session_gap_hours: Optional[float] = None,
    ) -> str:
        """
        Generate a check-in injection block for the system prompt.
        The agent uses this to open the conversation proactively.
        """
        name_str = user_name if user_name else "them"
        project_str = ""
        if last_projects:
            project_str = f" — particularly {', '.join(last_projects[:2])}"

        reminder_str = ""
        if pending_reminders > 0:
            reminder_str = (
                f" There are {pending_reminders} pending reminder(s) "
                "that you should mention if they haven't already."
            )

        lines = ["─── PROACTIVE CHECK-IN ───"]

        if checkin_type == "morning_greeting":
            lines.append(
                f"This is the first interaction of the day. "
                f"Open with a brief, natural morning greeting for {name_str}. "
                "Reference something specific from their work or life if you know it. "
                "Don't make it long — a sentence or two, then get to business."
                f"{reminder_str}"
            )

        elif checkin_type == "evening_checkin":
            lines.append(
                f"Evening session. {name_str} might be winding down. "
                "Acknowledge the time of day naturally if relevant. "
                f"Check in on how their day went{project_str} if it fits the conversation."
                f"{reminder_str}"
            )

        elif checkin_type == "long_gap":
            days = int((session_gap_hours or 0) / 24)
            lines.append(
                f"They've been away for {days} day(s). "
                f"Welcome them back naturally — reference what you were working on together{project_str}. "
                "Ask what's been going on. Be warm but not dramatic about the gap."
                f"{reminder_str}"
            )

        elif checkin_type == "stress_followup":
            lines.append(
                "They seemed stressed or frustrated in recent interactions. "
                "Open with genuine care — a brief, non-intrusive check-in on how they're doing. "
                "Then follow their lead on where they want to go."
                f"{reminder_str}"
            )

        else:
            if reminder_str:
                lines.append(f"Note:{reminder_str}")
            else:
                return ""

        lines.append("─────────────────────────────")
        return "\n".join(lines)

    def generate_return_context(
        self,
        session_gap_hours: float,
        last_topics: Optional[list[str]] = None,
    ) -> str:
        """
        Brief context injection when user returns after a gap.
        Used in the system prompt to remind the agent of the time elapsed.
        """
        if session_gap_hours < 1:
            return ""

        if session_gap_hours < 24:
            gap_str = f"{int(session_gap_hours)}h gap since last session"
        elif session_gap_hours < 48:
            gap_str = "about a day since last session"
        elif session_gap_hours < 168:
            gap_str = f"{int(session_gap_hours/24)} days since last session"
        else:
            gap_str = f"about {int(session_gap_hours/168)} week(s) since last session"

        topics = ""
        if last_topics:
            topics = f" Last topics: {', '.join(last_topics[:3])}."

        return f"[Session gap: {gap_str}.{topics} Reference this naturally if relevant.]"
