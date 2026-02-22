"""
Personality routes — GET/PATCH /api/personality, GET /api/notifications.

TARS-style personality control: adjust traits, view current calibration,
retrieve proactive notifications.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["personality"])


class TraitUpdate(BaseModel):
    trait: str
    value: float


class PersonalityUpdate(BaseModel):
    humor_level: float | None = None
    wit_level: float | None = None
    directness_level: float | None = None
    warmth_level: float | None = None
    sass_level: float | None = None
    formality_level: float | None = None
    proactive_level: float | None = None
    empathy_level: float | None = None
    curiosity_level: float | None = None


@router.get("/personality")
async def get_personality():
    """Return current personality trait values."""
    from psycho.api.server import agent
    if not agent or not agent.personality:
        return {"traits": {}, "status": "not_initialized"}
    traits = agent.personality.traits.to_dict()
    return {
        "traits": traits,
        "status_line": agent.personality.get_trait_status(),
    }


@router.patch("/personality")
async def update_personality(update: PersonalityUpdate):
    """
    Update personality traits.
    Values must be 0.0–1.0 (representing 0%–100%).

    Example:
        PATCH /api/personality
        { "humor_level": 0.9, "directness_level": 1.0 }
    """
    from psycho.api.server import agent
    from psycho.config import get_settings

    if not agent or not agent.personality:
        raise HTTPException(503, "Agent not initialized")

    changes = []
    update_dict = update.model_dump(exclude_none=True)

    for trait, value in update_dict.items():
        if not 0.0 <= value <= 1.0:
            raise HTTPException(400, f"Value for {trait} must be 0.0–1.0, got {value}")
        if agent.personality.traits.set_trait(trait, value):
            changes.append(f"{trait}: {int(value*100)}%")

    if changes:
        path = get_settings().get_personality_path()
        agent.personality.traits.save(path)

    return {
        "updated": changes,
        "traits": agent.personality.traits.to_dict(),
    }


@router.post("/personality/trait")
async def set_single_trait(body: TraitUpdate):
    """Set a single personality trait by name and value."""
    from psycho.api.server import agent
    from psycho.config import get_settings

    if not agent or not agent.personality:
        raise HTTPException(503, "Agent not initialized")

    if not 0.0 <= body.value <= 1.0:
        raise HTTPException(400, f"Value must be 0.0–1.0, got {body.value}")

    success = agent.personality.traits.set_trait(body.trait, body.value)
    if not success:
        raise HTTPException(400, f"Unknown trait: '{body.trait}'")

    path = get_settings().get_personality_path()
    agent.personality.traits.save(path)

    return {
        "trait": body.trait,
        "value": body.value,
        "traits": agent.personality.traits.to_dict(),
    }


# ── Notifications ──────────────────────────────────────────────────────────────


@router.get("/notifications")
async def get_notifications(unread_only: bool = False, limit: int = 20):
    """
    Get pending proactive notifications (due reminders, calendar alerts, check-ins).

    Poll this endpoint every 30 seconds from the frontend.
    """
    from psycho.api.server import agent

    if not agent or not agent.scheduler:
        return {"notifications": [], "unread_count": 0}

    if unread_only:
        notifs = agent.scheduler.get_unread()
    else:
        notifs = agent.scheduler.get_all(limit=limit)

    return {
        "notifications": [n.to_dict() for n in notifs],
        "unread_count": agent.scheduler.unread_count,
    }


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark a notification as read."""
    from psycho.api.server import agent

    if not agent or not agent.scheduler:
        raise HTTPException(503, "Scheduler not running")

    success = agent.scheduler.mark_read(notification_id)
    return {"success": success, "unread_count": agent.scheduler.unread_count}


@router.post("/notifications/read-all")
async def mark_all_read():
    """Mark all notifications as read."""
    from psycho.api.server import agent

    if not agent or not agent.scheduler:
        raise HTTPException(503, "Scheduler not running")

    count = agent.scheduler.mark_all_read()
    return {"marked_read": count}


# ── Reminders ──────────────────────────────────────────────────────────────────


class ReminderCreate(BaseModel):
    title: str
    due_timestamp: float
    notes: str = ""
    recurrence: str = "none"
    priority: str = "normal"


@router.get("/reminders")
async def get_reminders(pending_only: bool = True):
    """Get all reminders."""
    from psycho.api.server import agent

    if not agent or not agent.reminder_manager:
        return {"reminders": []}

    if pending_only:
        reminders = await agent.reminder_manager.get_all_pending()
    else:
        # Show all including completed
        reminders = await agent.reminder_manager.get_all_pending()

    return {"reminders": [r.to_dict() for r in reminders]}


@router.post("/reminders")
async def create_reminder(body: ReminderCreate):
    """Create a new reminder."""
    from psycho.api.server import agent

    if not agent or not agent.reminder_manager:
        raise HTTPException(503, "Agent not initialized")

    reminder = await agent.reminder_manager.create(
        title=body.title,
        due_timestamp=body.due_timestamp,
        notes=body.notes,
        recurrence=body.recurrence,
        priority=body.priority,
    )
    return reminder.to_dict()


@router.patch("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: str):
    """Mark a reminder as completed."""
    from psycho.api.server import agent

    if not agent or not agent.reminder_manager:
        raise HTTPException(503, "Agent not initialized")

    success = await agent.reminder_manager.complete(reminder_id)
    return {"success": success}


@router.patch("/reminders/{reminder_id}/snooze")
async def snooze_reminder(reminder_id: str, minutes: int = 15):
    """Snooze a reminder for N minutes."""
    from psycho.api.server import agent

    if not agent or not agent.reminder_manager:
        raise HTTPException(503, "Agent not initialized")

    success = await agent.reminder_manager.snooze(reminder_id, minutes=minutes)
    return {"success": success, "snoozed_minutes": minutes}


# ── Calendar ───────────────────────────────────────────────────────────────────


class CalendarEventCreate(BaseModel):
    title: str
    start_timestamp: float
    end_timestamp: float | None = None
    location: str = ""
    notes: str = ""
    recurrence: str = "none"
    all_day: bool = False
    reminder_minutes: int = 15


@router.get("/calendar")
async def get_calendar_events(days_ahead: float = 7.0):
    """Get upcoming calendar events."""
    from psycho.api.server import agent

    if not agent or not agent.calendar_manager:
        return {"events": []}

    events = await agent.calendar_manager.get_upcoming(hours_ahead=days_ahead * 24)
    return {"events": [e.to_dict() for e in events]}


@router.get("/calendar/today")
async def get_today_events():
    """Get today's calendar events."""
    from psycho.api.server import agent

    if not agent or not agent.calendar_manager:
        return {"events": []}

    events = await agent.calendar_manager.get_today()
    return {"events": [e.to_dict() for e in events]}


@router.post("/calendar")
async def create_calendar_event(body: CalendarEventCreate):
    """Create a new calendar event."""
    from psycho.api.server import agent

    if not agent or not agent.calendar_manager:
        raise HTTPException(503, "Agent not initialized")

    event = await agent.calendar_manager.add_event(
        title=body.title,
        start_timestamp=body.start_timestamp,
        end_timestamp=body.end_timestamp,
        location=body.location,
        notes=body.notes,
        recurrence=body.recurrence,
        all_day=body.all_day,
        reminder_minutes=body.reminder_minutes,
    )
    return event.to_dict()


@router.delete("/calendar/{event_id}")
async def delete_calendar_event(event_id: str):
    """Delete a calendar event."""
    from psycho.api.server import agent

    if not agent or not agent.calendar_manager:
        raise HTTPException(503, "Agent not initialized")

    success = await agent.calendar_manager.delete_event(event_id)
    return {"success": success}
