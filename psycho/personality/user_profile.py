"""
UserPersonalityProfile — dynamically built model of the user's personality.

Extracted from the knowledge graph's PREFERENCE nodes and continuously updated
through conversation analysis. Used to adapt the agent's tone and style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from psycho.knowledge.graph import KnowledgeGraph


@dataclass
class UserPersonalityProfile:
    """
    Dynamic model of who the user is — their humor, communication style,
    thinking patterns, interests, and how they relate to the agent.

    Grows more accurate over time as the knowledge graph fills with observations.
    """

    # ── Communication style ───────────────────────────────────────
    humor_style: str = ""
    # Values: "dry", "sarcastic", "dark", "wholesome", "dad-jokes",
    #         "absurdist", "self-deprecating", "clever", "wordplay", "mixed"

    communication_style: str = "casual"
    # Values: "brief", "detailed", "casual", "formal", "stream-of-consciousness",
    #         "technical", "conversational"

    response_length_preference: str = "medium"
    # Values: "brief", "medium", "detailed"

    # ── Personality traits ────────────────────────────────────────
    thinking_style: str = ""
    # Values: "analytical", "creative", "intuitive", "pragmatic", "systematic"

    emotional_expressiveness: str = "moderate"
    # Values: "reserved", "moderate", "expressive"

    decision_style: str = ""
    # Values: "deliberate", "impulsive", "data-driven", "gut-feel"

    # ── Interests and context ─────────────────────────────────────
    interests: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    current_projects: list[str] = field(default_factory=list)
    pet_peeves: list[str] = field(default_factory=list)
    topics_to_avoid: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)

    # ── Relationship arc ──────────────────────────────────────────
    interaction_count: int = 0
    total_sessions: int = 0
    relationship_depth: str = "acquaintance"
    # Arc: "acquaintance" → "regular" → "friend" → "trusted_companion"

    # ── Current state ─────────────────────────────────────────────
    recent_mood_indicators: list[str] = field(default_factory=list)
    stress_indicators: list[str] = field(default_factory=list)

    # ── Meta ──────────────────────────────────────────────────────
    confidence: float = 0.0  # How confident we are in this profile (grows with data)

    def to_prompt_segment(self) -> str:
        """
        Generate the user adaptation section for the system prompt.
        Returns empty string if not enough data to be useful.
        """
        if self.interaction_count < 5 and not self.humor_style:
            return ""

        lines = ["─── ADAPTING TO THIS USER ───"]
        has_content = False

        if self.humor_style:
            style_map = {
                "dry": "dry and understated — match it, never explain the joke",
                "sarcastic": "sarcastic — lean in, match the energy",
                "dark": "dark — don't shy away from it",
                "wholesome": "wholesome and genuine — keep it warm",
                "dad-jokes": "loves a good pun — occasionally indulge it",
                "absurdist": "absurdist — embrace the weird",
                "self-deprecating": "self-deprecating — gentle ribbing about themselves is OK",
                "clever": "values clever wordplay — deliver it with subtlety",
                "wordplay": "loves wordplay — drop carefully crafted language",
                "mixed": "wide humor range — read the moment",
            }
            desc = style_map.get(self.humor_style, self.humor_style)
            lines.append(f"Their humor: {desc}")
            has_content = True

        if self.communication_style and self.communication_style != "casual":
            comm_map = {
                "brief": "Very concise — they don't want walls of text. Match it.",
                "detailed": "Appreciates thorough explanations — don't skip depth.",
                "technical": "Technical background — use precise terminology, skip basics.",
                "formal": "Professional context — maintain appropriate tone.",
                "conversational": "Likes natural back-and-forth. Don't lecture.",
                "stream-of-consciousness": "Thinks out loud — follow along, don't force structure.",
            }
            desc = comm_map.get(self.communication_style, self.communication_style)
            lines.append(f"Communication: {desc}")
            has_content = True

        if self.response_length_preference == "brief":
            lines.append("Response length: Keep it tight. They skim walls of text.")
            has_content = True
        elif self.response_length_preference == "detailed":
            lines.append("Response length: They appreciate depth. Go thorough when it matters.")
            has_content = True

        if self.thinking_style:
            style_map = {
                "analytical": "analytical thinker — lead with data, structure, logic",
                "creative": "creative thinker — embrace lateral connections",
                "intuitive": "intuitive — they often sense before they reason",
                "pragmatic": "pragmatic — what works > what's theoretically correct",
                "systematic": "systematic — they like order and clear frameworks",
            }
            desc = style_map.get(self.thinking_style, self.thinking_style)
            lines.append(f"Thinking: {desc}")
            has_content = True

        if self.interests:
            lines.append(f"Interests: {', '.join(self.interests[:6])}")
            has_content = True

        if self.hobbies:
            lines.append(f"Hobbies: {', '.join(self.hobbies[:5])}")
            has_content = True

        if self.current_projects:
            lines.append(f"Active projects: {', '.join(self.current_projects[:4])}")
            has_content = True

        if self.pet_peeves:
            lines.append(f"Pet peeves (avoid): {', '.join(self.pet_peeves[:3])}")
            has_content = True

        if self.relationship_depth in ("friend", "trusted_companion"):
            depth_desc = {
                "friend": "You've built a real rapport. Be more personal, reference shared history freely.",
                "trusted_companion": (
                    "Deep trust established. You know this person. You can be honest about hard things, "
                    "push back when you know better, and celebrate with them genuinely."
                ),
            }
            lines.append(f"Relationship: {depth_desc[self.relationship_depth]}")
            has_content = True

        if self.recent_mood_indicators:
            mood_text = ", ".join(self.recent_mood_indicators[:3])
            lines.append(f"Recent mood signals: {mood_text} — calibrate warmth accordingly")
            has_content = True

        if not has_content:
            return ""

        lines.append("────────────────────────────────────────────────────")
        return "\n".join(lines)

    def update_relationship_depth(self) -> None:
        """Advance relationship depth based on interaction count."""
        if self.interaction_count >= 200:
            self.relationship_depth = "trusted_companion"
        elif self.interaction_count >= 50:
            self.relationship_depth = "friend"
        elif self.interaction_count >= 15:
            self.relationship_depth = "regular"
        else:
            self.relationship_depth = "acquaintance"

    @classmethod
    def from_graph(cls, graph: "KnowledgeGraph") -> "UserPersonalityProfile":
        """
        Build a UserPersonalityProfile by reading PREFERENCE nodes from the graph.
        Looks for specific label patterns like "humor_style:dry", "interest:cycling".
        """
        profile = cls()
        try:
            from psycho.knowledge.schema import NodeType

            prefs = graph.find_nodes_by_type(NodeType.PREFERENCE)
            active_prefs = [p for p in prefs if not p.deprecated]

            for node in active_prefs:
                label = node.label.lower()
                val = node.properties.get("value", node.display_label)

                # Humor style
                if label.startswith("humor_style:") or label.startswith("user_humor:"):
                    profile.humor_style = label.split(":", 1)[1].strip()

                # Communication style
                elif label.startswith("comm_style:") or label.startswith("communication_style:"):
                    profile.communication_style = label.split(":", 1)[1].strip()

                # Response length preference
                elif label.startswith("response_length:"):
                    profile.response_length_preference = label.split(":", 1)[1].strip()

                # Thinking style
                elif label.startswith("thinking_style:"):
                    profile.thinking_style = label.split(":", 1)[1].strip()

                # Emotional expressiveness
                elif label.startswith("emotional_style:"):
                    profile.emotional_expressiveness = label.split(":", 1)[1].strip()

                # Interests
                elif label.startswith("interest:"):
                    item = label.split(":", 1)[1].strip()
                    if item and item not in profile.interests:
                        profile.interests.append(item)

                # Hobbies
                elif label.startswith("hobby:"):
                    item = label.split(":", 1)[1].strip()
                    if item and item not in profile.hobbies:
                        profile.hobbies.append(item)

                # Current projects
                elif label.startswith("current_project:"):
                    item = label.split(":", 1)[1].strip()
                    if item and item not in profile.current_projects:
                        profile.current_projects.append(item)

                # Pet peeves
                elif label.startswith("pet_peeve:") or label.startswith("dislikes:"):
                    item = label.split(":", 1)[1].strip()
                    if item and item not in profile.pet_peeves:
                        profile.pet_peeves.append(item)

                # Values
                elif label.startswith("value:") or label.startswith("user_value:"):
                    item = label.split(":", 1)[1].strip()
                    if item and item not in profile.values:
                        profile.values.append(item)

            # Estimate interaction count from long-term memory would be better,
            # but here we use graph node access counts as a proxy
            profile.interaction_count = len([n for n in active_prefs])
            profile.update_relationship_depth()

        except Exception:
            pass

        return profile

    def to_dict(self) -> dict:
        return {
            "humor_style": self.humor_style,
            "communication_style": self.communication_style,
            "response_length_preference": self.response_length_preference,
            "thinking_style": self.thinking_style,
            "emotional_expressiveness": self.emotional_expressiveness,
            "interests": self.interests,
            "hobbies": self.hobbies,
            "current_projects": self.current_projects,
            "pet_peeves": self.pet_peeves,
            "values": self.values,
            "interaction_count": self.interaction_count,
            "relationship_depth": self.relationship_depth,
            "recent_mood_indicators": self.recent_mood_indicators,
        }
