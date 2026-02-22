"""Personality engine — TARS/Jarvis-style adjustable personality system."""

from .adapter import PersonalityAdapter
from .traits import AgentPersonality
from .user_profile import UserPersonalityProfile

__all__ = ["AgentPersonality", "UserPersonalityProfile", "PersonalityAdapter"]
