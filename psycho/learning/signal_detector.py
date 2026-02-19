"""
Signal detector — identifies corrections, confirmations, and feedback signals
in user messages in real-time, before the LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SignalType(str, Enum):
    NONE         = "none"
    CORRECTION   = "correction"     # User is correcting the agent
    CONFIRMATION = "confirmation"   # User is confirming something correct
    FRUSTRATION  = "frustration"    # User is frustrated (low quality signal)
    QUESTION     = "question"       # Neutral question, no feedback


@dataclass
class Signal:
    type: SignalType
    confidence: float           # 0.0 – 1.0 how sure we are
    snippet: str = ""           # Matched text that triggered detection
    correction_hint: str = ""   # What the correct info might be (if extractable)


# ── Pattern banks ──────────────────────────────────────────────────────────────

_STRONG_CORRECTION = re.compile(
    r"\b("
    r"wrong|incorrect|not right|not true|that'?s (not|wrong|incorrect)|"
    r"you'?re wrong|you (are|were) (wrong|incorrect|mistaken)|"
    r"that is (wrong|incorrect|not right|false)|"
    r"actually[,\s]|actually it'?s|actually that'?s|"
    r"no[,!]\s|nope[,!]?\s|that'?s not|it'?s not|it is not|"
    r"correction:|wrong:|mistake:|fix:|no[,]? the (correct|right|actual)"
    r")",
    re.IGNORECASE,
)

_MODERATE_CORRECTION = re.compile(
    r"\b("
    r"should be|the (correct|right|actual|proper) (answer|version|value|way) is|"
    r"you (meant|said) .{0,30} but it'?s|"
    r"not .{0,20} but .{0,20}|"
    r"i think you.{0,20}wrong|"
    r"that'?s (a )?mistake|"
    r"let me (correct|clarify|fix)"
    r")",
    re.IGNORECASE,
)

_STRONG_CONFIRMATION = re.compile(
    r"\b("
    r"yes[,!]?\s|yeah[,!]?\s|yep[,!]?\s|yup[,!]?\s|"
    r"correct[,!]?(\s|$)|right[,!]?(\s|$)|exactly[,!]?(\s|$)|"
    r"that'?s (right|correct|exactly|it|perfect)|"
    r"you'?re right|you (are|were) right|"
    r"perfect|exactly right|spot on|precisely|"
    r"good (job|answer|response|point)|"
    r"(that|this) is (correct|right|accurate)"
    r")",
    re.IGNORECASE,
)

_FRUSTRATION = re.compile(
    r"\b("
    r"this is (useless|terrible|bad|awful|wrong)|"
    r"you keep (getting|making|saying)|"
    r"how (many|much) times|"
    r"(not )?again[!?]|"
    r"come on[!?]|seriously[!?]"
    r")",
    re.IGNORECASE,
)


def detect_signal(user_message: str) -> Signal:
    """
    Detect feedback signals in a user message.

    Returns a Signal with the detected type and confidence.
    Fast, regex-only — no LLM call needed here.
    """
    msg = user_message.strip()

    # Very short messages are usually confirmations or simple replies
    if len(msg) < 4:
        return Signal(type=SignalType.NONE, confidence=0.0)

    # Check strong correction first (highest priority)
    m = _STRONG_CORRECTION.search(msg)
    if m:
        return Signal(
            type=SignalType.CORRECTION,
            confidence=0.85,
            snippet=m.group(0),
        )

    # Moderate correction
    m = _MODERATE_CORRECTION.search(msg)
    if m:
        return Signal(
            type=SignalType.CORRECTION,
            confidence=0.65,
            snippet=m.group(0),
        )

    # Strong confirmation
    m = _STRONG_CONFIRMATION.search(msg)
    if m:
        # Make sure it's not a negated confirmation ("no, that's not right")
        if not _STRONG_CORRECTION.search(msg):
            return Signal(
                type=SignalType.CONFIRMATION,
                confidence=0.75,
                snippet=m.group(0),
            )

    # Frustration signal
    m = _FRUSTRATION.search(msg)
    if m:
        return Signal(
            type=SignalType.FRUSTRATION,
            confidence=0.6,
            snippet=m.group(0),
        )

    return Signal(type=SignalType.NONE, confidence=0.0)


def extract_correction_target(user_message: str, recent_agent_response: str) -> str:
    """
    Try to identify what was being corrected from context.
    Returns a hint string (may be empty if unclear).
    """
    # Look for patterns like "X is Y" or "it's actually X"
    patterns = [
        r"actually[,\s]+(.{10,100}?)(?:[.!?]|$)",
        r"it'?s\s+(.{5,80}?)(?:[.!?]|$)",
        r"the (?:correct|right|actual) (?:answer|value|version) is\s+(.{5,80}?)(?:[.!?]|$)",
        r"should be\s+(.{5,80}?)(?:[.!?]|$)",
        r"not .{0,30}? but\s+(.{5,80}?)(?:[.!?]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, user_message, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""
