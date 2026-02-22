"""
PersonalityAdapter — generates dynamic personality-aware system prompt sections.

Combines AgentPersonality (how the agent behaves) with UserPersonalityProfile
(who the user is) to produce prompt injections that make the agent feel
genuinely calibrated and human-like.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from .traits import AgentPersonality, detect_trait_command
from .user_profile import UserPersonalityProfile

if TYPE_CHECKING:
    from psycho.knowledge.graph import KnowledgeGraph


# Patterns that detect mood signals in user messages
STRESS_PATTERNS = [
    r"\b(stressed|overwhelmed|anxious|anxious|frustrated|stuck|lost|confused|struggling)\b",
    r"\b(ugh|argh|ffs|wtf|damn|dammit|exhausted|burnt out|burnout)\b",
    r"(don't know what|have no idea|can't figure|nothing works)",
]
EXCITEMENT_PATTERNS = [
    r"\b(excited|amazing|awesome|great news|just|finally|finally!|nailed it|got it working)\b",
    r"(!!+|😄|🎉|🔥|🚀)",
]
TIRED_PATTERNS = [
    r"\b(tired|sleepy|up late|stayed up|exhausted|no sleep|barely slept)\b",
]


class PersonalityAdapter:
    """
    Generates and manages the personality sections injected into every system prompt.

    Responsibilities:
    - Generates agent personality calibration block (trait-based)
    - Generates user adaptation block (graph-derived user profile)
    - Detects personality change commands ("set humor to 90%")
    - Detects user mood shifts and adjusts emphasis
    - Saves/loads personality state from disk
    """

    def __init__(
        self,
        traits: AgentPersonality,
        graph: Optional["KnowledgeGraph"] = None,
        personality_path: Optional[Path] = None,
    ) -> None:
        self._traits = traits
        self._graph = graph
        self._personality_path = personality_path
        self._user_profile: Optional[UserPersonalityProfile] = None
        self._interaction_count: int = 0
        self._session_count: int = 0

    @property
    def traits(self) -> AgentPersonality:
        return self._traits

    def set_graph(self, graph: "KnowledgeGraph") -> None:
        self._graph = graph
        self._user_profile = None  # Invalidate cache

    def increment_interaction(self) -> None:
        self._interaction_count += 1
        # Refresh user profile every 5 interactions
        if self._interaction_count % 5 == 0:
            self._refresh_user_profile()

    def increment_session(self) -> None:
        self._session_count += 1

    def build_prompt_sections(
        self,
        user_message: str = "",
        conversation_length: int = 0,
    ) -> tuple[str, str]:
        """
        Build (personality_section, user_adaptation_section) for system prompt.

        Args:
            user_message: Current user message (used for mood detection)
            conversation_length: Number of turns in current conversation

        Returns:
            Tuple of (personality_block, user_adaptation_block)
        """
        personality_block = self._traits.to_prompt_segment()

        # Build/update user profile from graph
        if self._user_profile is None:
            self._refresh_user_profile()

        # Update interaction count in profile for relationship depth
        if self._user_profile:
            self._user_profile.interaction_count = self._interaction_count
            self._user_profile.total_sessions = self._session_count
            self._user_profile.update_relationship_depth()

            # Inject mood signals if detected
            if user_message:
                mood = self._detect_mood(user_message)
                if mood:
                    self._user_profile.recent_mood_indicators = [mood]
                else:
                    self._user_profile.recent_mood_indicators = []

        user_block = self._user_profile.to_prompt_segment() if self._user_profile else ""

        return personality_block, user_block

    def process_trait_command(self, user_message: str) -> list[str]:
        """
        Detect and apply personality trait commands from user message.

        Returns a list of human-readable change descriptions.
        Example return: ["Humor set to 90%", "Directness adjusted to 65%"]
        """
        commands = detect_trait_command(user_message)
        changes = []

        for cmd in commands:
            if len(cmd) == 2:
                # Direct value set: (trait, value)
                trait, value = cmd
                old = self._traits.get_trait(trait) or 0.0
                if self._traits.set_trait(trait, value):
                    display_name = trait.replace("_level", "").title()
                    changes.append(
                        f"{display_name} adjusted: {int(old*100)}% → {int(value*100)}%"
                    )
            elif len(cmd) == 3:
                # Delta: (trait, None, delta)
                trait, _, delta = cmd
                old = self._traits.get_trait(trait) or 0.0
                if self._traits.adjust_trait(trait, delta):
                    new_val = self._traits.get_trait(trait) or 0.0
                    display_name = trait.replace("_level", "").title()
                    direction = "↑" if delta > 0 else "↓"
                    changes.append(
                        f"{display_name} {direction}: {int(old*100)}% → {int(new_val*100)}%"
                    )

        if changes and self._personality_path:
            self._traits.save(self._personality_path)
            logger.info(f"Personality saved after adjustments: {changes}")

        return changes

    def is_trait_command(self, user_message: str) -> bool:
        """Quick check if the message appears to contain a personality command."""
        msg = user_message.lower()
        trait_words = [
            "humor", "humour", "wit", "direct", "warm", "sass", "formal",
            "proactive", "empathy", "personality", "calibrat",
        ]
        command_words = ["set", "dial", "turn", "be more", "be less", "adjust"]
        has_trait = any(t in msg for t in trait_words)
        has_cmd = any(c in msg for c in command_words)
        has_pct = bool(re.search(r"\d+%", msg))
        return has_trait and (has_cmd or has_pct)

    def get_trait_status(self) -> str:
        """Return formatted trait status for display."""
        return self._traits.format_status()

    # ── Private helpers ────────────────────────────────────────────

    def _refresh_user_profile(self) -> None:
        if self._graph:
            try:
                self._user_profile = UserPersonalityProfile.from_graph(self._graph)
            except Exception as e:
                logger.debug(f"User profile refresh failed: {e}")
                if self._user_profile is None:
                    self._user_profile = UserPersonalityProfile()
        else:
            if self._user_profile is None:
                self._user_profile = UserPersonalityProfile()

    def _detect_mood(self, message: str) -> Optional[str]:
        """Detect emotional signals in the user's message."""
        msg = message.lower()
        for pattern in STRESS_PATTERNS:
            if re.search(pattern, msg):
                return "stressed/frustrated"
        for pattern in EXCITEMENT_PATTERNS:
            if re.search(pattern, msg):
                return "excited/energized"
        for pattern in TIRED_PATTERNS:
            if re.search(pattern, msg):
                return "tired/low energy"
        return None

    @classmethod
    def create(
        cls,
        personality_path: Optional[Path] = None,
        graph: Optional["KnowledgeGraph"] = None,
    ) -> "PersonalityAdapter":
        """Factory: load existing personality or create defaults."""
        if personality_path and personality_path.exists():
            traits = AgentPersonality.load(personality_path)
            logger.info(f"Personality loaded from {personality_path}")
        else:
            traits = AgentPersonality()
            logger.info("Default personality initialized")

        return cls(traits=traits, graph=graph, personality_path=personality_path)
