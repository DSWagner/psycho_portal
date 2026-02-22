"""
ProactiveScheduler — async background task that monitors reminders and calendar events.

Runs as a background asyncio task inside the FastAPI server.
Every 60 seconds it checks:
  - Due reminders → sends notification
  - Calendar events needing pre-event alerts → sends notification
  - Check-in conditions → queues a proactive message

Notifications are stored in-memory and retrieved via /api/notifications polling endpoint.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from loguru import logger


@dataclass
class Notification:
    """A proactive notification queued for delivery to the UI."""

    id: str = ""
    type: str = ""         # "reminder", "calendar", "checkin", "task_due"
    title: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: str = "normal"  # "low" | "normal" | "high" | "urgent"
    action: str = ""       # "complete_reminder:{id}", "open_chat", etc.
    read: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "action": self.action,
            "read": self.read,
            "time_str": datetime.fromtimestamp(self.timestamp).strftime("%H:%M"),
        }


class ProactiveScheduler:
    """
    Background async loop that generates proactive notifications.

    Designed to run alongside the FastAPI server as a fire-and-forget task.
    Notifications accumulate in a bounded deque and are served via polling.
    """

    TICK_INTERVAL = 60  # Seconds between checks
    MAX_NOTIFICATIONS = 50  # Keep last N notifications

    def __init__(
        self,
        reminder_manager=None,
        calendar_manager=None,
        checkin_engine=None,
        on_notification: Optional[Callable] = None,
    ) -> None:
        self._reminders = reminder_manager
        self._calendar = calendar_manager
        self._checkin = checkin_engine
        self._on_notification = on_notification  # Optional callback (e.g., WebSocket push)

        self._notifications: deque[Notification] = deque(maxlen=self.MAX_NOTIFICATIONS)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._notified_ids: set[str] = set()  # Prevent duplicate notifications
        self._notification_counter = 0

    async def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("ProactiveScheduler started")

    async def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ProactiveScheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}")
            await asyncio.sleep(self.TICK_INTERVAL)

    async def _tick(self) -> None:
        """Single scheduler tick — check all proactive conditions."""
        await self._check_reminders()
        await self._check_calendar()

    async def _check_reminders(self) -> None:
        """Check for due reminders and create notifications."""
        if not self._reminders:
            return
        try:
            due = await self._reminders.get_due()
            for reminder in due:
                notify_key = f"reminder_{reminder.id}"
                if notify_key in self._notified_ids:
                    continue

                self._notified_ids.add(notify_key)
                self._emit(Notification(
                    id=self._next_id(),
                    type="reminder",
                    title=f"⏰ Reminder: {reminder.title}",
                    message=(
                        f"This was due at {reminder.due_dt.strftime('%H:%M')}."
                        + (f" Notes: {reminder.notes}" if reminder.notes else "")
                    ),
                    priority=reminder.priority,
                    action=f"complete_reminder:{reminder.id}",
                ))

                # Auto-reschedule recurring reminders
                if reminder.recurrence != "none":
                    await self._reminders.reschedule_recurring(reminder)

        except Exception as e:
            logger.debug(f"Reminder check failed: {e}")

    async def _check_calendar(self) -> None:
        """Check for calendar events needing pre-event alerts."""
        if not self._calendar:
            return
        try:
            events = await self._calendar.get_needing_reminder()
            for evt in events:
                notify_key = f"cal_{evt.id}_{int(evt.start_timestamp)}"
                if notify_key in self._notified_ids:
                    continue

                self._notified_ids.add(notify_key)
                time_str = evt.time_until_start()
                location_str = f" @ {evt.location}" if evt.location else ""
                self._emit(Notification(
                    id=self._next_id(),
                    type="calendar",
                    title=f"📅 Coming up: {evt.title}",
                    message=f"Starts in {time_str}{location_str}.",
                    priority="high",
                    action=f"open_calendar:{evt.id}",
                ))
        except Exception as e:
            logger.debug(f"Calendar check failed: {e}")

    def _emit(self, notification: Notification) -> None:
        """Add a notification to the queue and trigger callback if set."""
        self._notifications.append(notification)
        logger.info(f"Notification: [{notification.type}] {notification.title}")
        if self._on_notification:
            try:
                self._on_notification(notification)
            except Exception as e:
                logger.debug(f"Notification callback error: {e}")

    def _next_id(self) -> str:
        self._notification_counter += 1
        return f"notif_{self._notification_counter}"

    # ── Public API (for polling endpoint) ─────────────────────────

    def get_unread(self) -> list[Notification]:
        """Return all unread notifications."""
        return [n for n in self._notifications if not n.read]

    def get_all(self, limit: int = 20) -> list[Notification]:
        """Return the most recent N notifications."""
        all_notifs = list(self._notifications)
        return all_notifs[-limit:]

    def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def mark_all_read(self) -> int:
        count = 0
        for n in self._notifications:
            if not n.read:
                n.read = True
                count += 1
        return count

    def add_manual(
        self,
        title: str,
        message: str,
        notif_type: str = "info",
        priority: str = "normal",
    ) -> Notification:
        """Manually inject a notification (e.g., from a background task)."""
        n = Notification(
            id=self._next_id(),
            type=notif_type,
            title=title,
            message=message,
            priority=priority,
        )
        self._emit(n)
        return n

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def unread_count(self) -> int:
        return sum(1 for n in self._notifications if not n.read)
