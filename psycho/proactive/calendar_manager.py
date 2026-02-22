"""
CalendarManager — local calendar with optional Google Calendar sync.

Events are stored locally in SQLite. Google Calendar integration is optional
and activated by setting GOOGLE_CALENDAR_CREDENTIALS in .env.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger


@dataclass
class CalendarEvent:
    """A calendar event."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    location: str = ""
    notes: str = ""
    recurrence: str = "none"   # "none" | "daily" | "weekly" | "monthly" | "yearly"
    google_event_id: str = ""  # Set when synced to Google Calendar
    all_day: bool = False
    reminder_minutes: int = 15  # Notify N minutes before
    created_at: float = field(default_factory=time.time)

    @property
    def start_dt(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def end_dt(self) -> datetime:
        return datetime.fromtimestamp(self.end_timestamp)

    @property
    def duration_minutes(self) -> int:
        return max(0, int((self.end_timestamp - self.start_timestamp) / 60))

    def is_today(self) -> bool:
        today = datetime.now().date()
        return self.start_dt.date() == today

    def is_upcoming(self, hours: float = 24.0) -> bool:
        now = time.time()
        return now <= self.start_timestamp <= now + hours * 3600

    def needs_reminder(self) -> bool:
        """Returns True if it's time to send the pre-event reminder."""
        now = time.time()
        remind_at = self.start_timestamp - self.reminder_minutes * 60
        return remind_at <= now < self.start_timestamp

    def time_until_start(self) -> str:
        delta = self.start_timestamp - time.time()
        if delta < 0:
            return "started"
        if delta < 60:
            return "< 1 minute"
        if delta < 3600:
            return f"{int(delta/60)} min"
        hours = int(delta / 3600)
        return f"{hours}h"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "location": self.location,
            "notes": self.notes,
            "recurrence": self.recurrence,
            "google_event_id": self.google_event_id,
            "all_day": self.all_day,
            "reminder_minutes": self.reminder_minutes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row) -> "CalendarEvent":
        return cls(
            id=row["id"],
            title=row["title"],
            start_timestamp=row["start_timestamp"],
            end_timestamp=row["end_timestamp"],
            location=row["location"] or "",
            notes=row["notes"] or "",
            recurrence=row["recurrence"] or "none",
            google_event_id=row["google_event_id"] or "",
            all_day=bool(row["all_day"]),
            reminder_minutes=row["reminder_minutes"] or 15,
            created_at=row["created_at"],
        )


class CalendarManager:
    """
    Manages calendar events in SQLite with optional Google Calendar sync.

    Local-first: all events stored locally, Google Calendar is an optional mirror.
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS calendar_events (
        id                  TEXT    PRIMARY KEY,
        title               TEXT    NOT NULL,
        start_timestamp     REAL    NOT NULL,
        end_timestamp       REAL    NOT NULL,
        location            TEXT    DEFAULT '',
        notes               TEXT    DEFAULT '',
        recurrence          TEXT    DEFAULT 'none',
        google_event_id     TEXT    DEFAULT '',
        all_day             INTEGER DEFAULT 0,
        reminder_minutes    INTEGER DEFAULT 15,
        created_at          REAL    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_calendar_start ON calendar_events(start_timestamp);
    """

    def __init__(self, db, google_credentials_path: Optional[str] = None) -> None:
        self._db = db
        self._google_creds_path = google_credentials_path
        self._gcal_service = None  # Lazy-loaded

    async def ensure_schema(self) -> None:
        await self._db.conn.executescript(self.SCHEMA_SQL)
        await self._db.conn.commit()

    async def add_event(
        self,
        title: str,
        start_timestamp: float,
        end_timestamp: Optional[float] = None,
        location: str = "",
        notes: str = "",
        recurrence: str = "none",
        all_day: bool = False,
        reminder_minutes: int = 15,
    ) -> CalendarEvent:
        """Add a new calendar event."""
        if end_timestamp is None:
            end_timestamp = start_timestamp + 3600  # Default 1 hour

        event = CalendarEvent(
            title=title,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            location=location,
            notes=notes,
            recurrence=recurrence,
            all_day=all_day,
            reminder_minutes=reminder_minutes,
        )

        await self._db.execute(
            """INSERT INTO calendar_events
               (id, title, start_timestamp, end_timestamp, location, notes,
                recurrence, google_event_id, all_day, reminder_minutes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?)""",
            (
                event.id, event.title, event.start_timestamp, event.end_timestamp,
                event.location, event.notes, event.recurrence,
                int(event.all_day), event.reminder_minutes, event.created_at,
            ),
        )
        logger.info(f"Calendar event added: '{title}' at {event.start_dt}")

        # Optionally sync to Google Calendar
        if self._gcal_service:
            try:
                await self._sync_to_google(event)
            except Exception as e:
                logger.warning(f"Google Calendar sync failed: {e}")

        return event

    async def get_today(self) -> list[CalendarEvent]:
        """Return all events for today."""
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time()).timestamp()
        end = start + 86400
        rows = await self._db.fetch_all(
            "SELECT * FROM calendar_events WHERE start_timestamp BETWEEN ? AND ? ORDER BY start_timestamp",
            (start, end),
        )
        return [CalendarEvent.from_row(r) for r in rows]

    async def get_upcoming(self, hours_ahead: float = 24.0) -> list[CalendarEvent]:
        """Return events starting within the next N hours."""
        now = time.time()
        cutoff = now + hours_ahead * 3600
        rows = await self._db.fetch_all(
            "SELECT * FROM calendar_events WHERE start_timestamp BETWEEN ? AND ? ORDER BY start_timestamp",
            (now, cutoff),
        )
        return [CalendarEvent.from_row(r) for r in rows]

    async def get_needing_reminder(self) -> list[CalendarEvent]:
        """Events whose pre-event reminder window is now."""
        upcoming = await self.get_upcoming(hours_ahead=1.0)
        return [e for e in upcoming if e.needs_reminder()]

    async def get_week_ahead(self) -> list[CalendarEvent]:
        return await self.get_upcoming(hours_ahead=168.0)

    async def delete_event(self, event_id: str) -> bool:
        cursor = await self._db.execute(
            "DELETE FROM calendar_events WHERE id = ?", (event_id,)
        )
        return cursor.rowcount > 0

    async def get_stats(self) -> dict:
        now = time.time()
        row = await self._db.fetch_one(
            "SELECT COUNT(*) FROM calendar_events WHERE start_timestamp > ?", (now,)
        )
        return {"upcoming_events": row[0] if row else 0}

    def format_for_prompt(self, events: list[CalendarEvent], label: str = "UPCOMING") -> str:
        """Format events for injection into system prompt context."""
        if not events:
            return ""
        lines = [f"─── {label} CALENDAR EVENTS ───"]
        for evt in events[:8]:
            dt_str = evt.start_dt.strftime("%a %b %d, %H:%M")
            duration = f" ({evt.duration_minutes}min)" if evt.duration_minutes else ""
            location = f" @ {evt.location}" if evt.location else ""
            time_str = evt.time_until_start()
            lines.append(f"• {evt.title} — {dt_str}{duration}{location} [in {time_str}]")
            if evt.notes:
                lines.append(f"  Notes: {evt.notes}")
        lines.append("─────────────────────────────")
        return "\n".join(lines)

    async def try_init_google(self, credentials_path: str) -> bool:
        """
        Attempt to initialize Google Calendar API.
        Returns True if successful, False if not configured or auth failed.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/calendar"]
            import json
            from pathlib import Path

            token_path = Path(credentials_path).parent / "token.json"
            creds = None

            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                token_path.write_text(creds.to_json())

            self._gcal_service = build("calendar", "v3", credentials=creds)
            logger.info("Google Calendar initialized")
            return True

        except ImportError:
            logger.debug("google-api-python-client not installed; Google Calendar disabled")
            return False
        except Exception as e:
            logger.warning(f"Google Calendar init failed: {e}")
            return False

    async def _sync_to_google(self, event: CalendarEvent) -> None:
        """Sync a local event to Google Calendar."""
        if not self._gcal_service:
            return

        body = {
            "summary": event.title,
            "location": event.location,
            "description": event.notes,
            "start": {"dateTime": event.start_dt.isoformat(), "timeZone": "local"},
            "end": {"dateTime": event.end_dt.isoformat(), "timeZone": "local"},
            "reminders": {
                "useDefault": False,
                "overrides": [{"method": "popup", "minutes": event.reminder_minutes}],
            },
        }

        result = self._gcal_service.events().insert(calendarId="primary", body=body).execute()
        google_id = result.get("id", "")
        if google_id:
            await self._db.execute(
                "UPDATE calendar_events SET google_event_id = ? WHERE id = ?",
                (google_id, event.id),
            )
            logger.info(f"Event synced to Google Calendar: {google_id}")
