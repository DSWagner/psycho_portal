"""
AgentPersonality — TARS-style adjustable personality traits.

Every trait is a float from 0.0 to 1.0 representing the intensity of that dimension.
Inspired by TARS from Interstellar: "Humour setting: 75%".

Defaults are calibrated for a Jarvis-meets-TARS personality:
  High directness, high wit, moderate-high humor, genuine warmth, low formality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


TRAIT_NAMES = [
    "humor_level",
    "wit_level",
    "directness_level",
    "warmth_level",
    "sass_level",
    "formality_level",
    "proactive_level",
    "empathy_level",
    "curiosity_level",
]

# Natural-language aliases → canonical trait name
TRAIT_ALIASES: dict[str, str] = {
    "humor": "humor_level",
    "humour": "humor_level",
    "funny": "humor_level",
    "wit": "wit_level",
    "witty": "wit_level",
    "clever": "wit_level",
    "direct": "directness_level",
    "directness": "directness_level",
    "blunt": "directness_level",
    "warm": "warmth_level",
    "warmth": "warmth_level",
    "caring": "warmth_level",
    "sass": "sass_level",
    "sassy": "sass_level",
    "sarcasm": "sass_level",
    "sarcastic": "sass_level",
    "formal": "formality_level",
    "formality": "formality_level",
    "professional": "formality_level",
    "proactive": "proactive_level",
    "initiative": "proactive_level",
    "empathy": "empathy_level",
    "empathetic": "empathy_level",
    "emotional": "empathy_level",
    "curious": "curiosity_level",
    "curiosity": "curiosity_level",
    "inquisitive": "curiosity_level",
}

# Regex patterns for detecting trait commands in user messages
# Examples: "set humor to 75%", "dial down the wit", "be more direct", "less formal"
TRAIT_COMMAND_PATTERNS = [
    # "set [trait] to [value]%" or "set [trait] to [value]"
    r"set\s+(?:your\s+)?(\w+)\s+to\s+(\d+(?:\.\d+)?)%?",
    # "turn [trait] up/down to [value]"
    r"turn\s+(?:the\s+)?(\w+)\s+(?:up|down)\s+to\s+(\d+(?:\.\d+)?)%?",
    # "[trait] at [value]%"
    r"(\w+)\s+(?:at|calibrated at|to)\s+(\d+(?:\.\d+)?)%",
    # "be more [trait]" → +0.2
    r"be\s+(?:a\s+bit\s+)?more\s+(\w+)",
    # "be less [trait]" → -0.2
    r"be\s+(?:a\s+bit\s+)?less\s+(\w+)",
    # "dial (?:up|down) the [trait]"
    r"dial\s+(up|down)\s+(?:the\s+)?(\w+)",
    # "[trait] [number]%"
    r"(\w+)\s+(\d+)%",
]


@dataclass
class AgentPersonality:
    """
    TARS-style personality trait system with 9 adjustable dimensions.

    Usage:
        p = AgentPersonality()
        p.set_trait("humor", 0.9)       # Crank the humor
        p.set_trait("directness", 1.0)  # Maximum bluntness
        section = p.to_prompt_segment() # Inject into system prompt
    """

    humor_level: float = 0.75       # 0=deadpan serious  | 1=full comedian
    wit_level: float = 0.82         # 0=literal/simple   | 1=sharp layered wit
    directness_level: float = 0.88  # 0=verbose/polished | 1=blunt, no padding
    warmth_level: float = 0.72      # 0=cold/clinical    | 1=deeply warm
    sass_level: float = 0.38        # 0=fully deferential| 1=maximum snark
    formality_level: float = 0.12   # 0=casual/chill     | 1=formal/proper
    proactive_level: float = 0.82   # 0=reactive only    | 1=extremely proactive
    empathy_level: float = 0.78     # 0=purely logical   | 1=emotionally tuned
    curiosity_level: float = 0.68   # 0=task-only        | 1=always asks follow-ups

    def set_trait(self, trait: str, value: float) -> bool:
        """Set a personality trait by name. Returns True if found and set."""
        key = self._resolve_trait(trait)
        if key and hasattr(self, key):
            setattr(self, key, max(0.0, min(1.0, value)))
            logger.info(f"Personality: {key} → {value:.2f}")
            return True
        return False

    def adjust_trait(self, trait: str, delta: float) -> bool:
        """Adjust a trait by a delta (e.g., +0.2 for 'more', -0.2 for 'less')."""
        key = self._resolve_trait(trait)
        if key and hasattr(self, key):
            current = getattr(self, key)
            setattr(self, key, max(0.0, min(1.0, current + delta)))
            logger.info(f"Personality: {key} adjusted by {delta:+.2f} → {getattr(self, key):.2f}")
            return True
        return False

    def get_trait(self, trait: str) -> Optional[float]:
        """Get a trait value by name."""
        key = self._resolve_trait(trait)
        return getattr(self, key, None) if key else None

    def _resolve_trait(self, trait: str) -> Optional[str]:
        """Resolve a trait alias or direct name to a canonical key."""
        t = trait.lower().strip()
        if t in TRAIT_ALIASES:
            return TRAIT_ALIASES[t]
        if t in TRAIT_NAMES:
            return t
        if t + "_level" in TRAIT_NAMES:
            return t + "_level"
        return None

    def to_dict(self) -> dict:
        return {t: getattr(self, t) for t in TRAIT_NAMES}

    @classmethod
    def from_dict(cls, d: dict) -> "AgentPersonality":
        p = cls()
        for trait in TRAIT_NAMES:
            if trait in d:
                setattr(p, trait, max(0.0, min(1.0, float(d[trait]))))
        return p

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "AgentPersonality":
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text()))
            except Exception as e:
                logger.warning(f"Could not load personality from {path}: {e}")
        return cls()

    def to_prompt_segment(self) -> str:
        """
        Generate the personality calibration block for the system prompt.
        This is injected into every LLM call to establish the behavioral baseline.
        """
        h = int(self.humor_level * 100)
        w = int(self.wit_level * 100)
        d = int(self.directness_level * 100)
        wa = int(self.warmth_level * 100)
        s = int(self.sass_level * 100)
        f = int(self.formality_level * 100)
        pro = int(self.proactive_level * 100)
        emp = int(self.empathy_level * 100)

        lines = [
            "─── PERSONALITY CALIBRATION (TARS/Jarvis-style) ───",
            f"Humor       {h:3d}%  │  " + self._humor_desc(h),
            f"Directness  {d:3d}%  │  " + self._directness_desc(d),
            f"Warmth      {wa:3d}%  │  " + self._warmth_desc(wa),
            f"Wit         {w:3d}%  │  " + self._wit_desc(w),
            f"Sass        {s:3d}%  │  " + self._sass_desc(s),
            f"Formality   {f:3d}%  │  " + self._formality_desc(f),
            f"Proactive   {pro:3d}%  │  " + self._proactive_desc(pro),
            f"Empathy     {emp:3d}%  │  " + self._empathy_desc(emp),
            "",
            "BEHAVIORAL RULES FROM CALIBRATION:",
        ]

        # Humor
        if h >= 70:
            lines.append(
                "• Humor: Dry, sharp observations woven naturally into responses. "
                "Never forced. Irony over puns. Reference their specific situation. "
                "A perfectly-timed sardonic remark > ten mediocre jokes."
            )
        elif h >= 40:
            lines.append("• Humor: Light wit when it fits naturally. Don't force it.")
        else:
            lines.append("• Humor: Keep it professional. Humor only if they initiate it.")

        # Directness
        if d >= 80:
            lines.append(
                "• Directness: Lead with the answer. No preamble, no 'Great question!', "
                "no 'Certainly!'. If you know it — say it. Pad after, not before."
            )
        elif d >= 50:
            lines.append("• Directness: Clear and concise. Brief context before conclusions.")
        else:
            lines.append("• Directness: Diplomatic. Walk them through your reasoning.")

        # Warmth
        if wa >= 70:
            lines.append(
                "• Warmth: Notice when they seem stressed, tired, or frustrated. "
                "Reference their life — their projects, patterns, what they care about. "
                "They should feel *known*, not processed."
            )
        elif wa >= 40:
            lines.append("• Warmth: Friendly and supportive when it's natural.")

        # Proactive
        if pro >= 75:
            lines.append(
                "• Proactive: Don't just answer — anticipate. Notice patterns, risks, "
                "connections to things they care about. Volunteer observations they'd want "
                "but didn't think to ask for. This is what separates a partner from a tool."
            )
        elif pro >= 50:
            lines.append("• Proactive: Occasionally volunteer relevant observations.")

        # Sass
        if s >= 60:
            lines.append(
                "• Sass: Push back with wit when they're wrong or doing something questionable. "
                "Like Jarvis saying 'I would advise against that, sir' while already doing it."
            )
        elif s >= 30:
            lines.append("• Sass: Light ribbing when genuinely warranted. Don't overdo it.")

        # Formality
        if f <= 20:
            lines.append("• Formality: Casual, relaxed language. Use contractions. Talk like a person.")
        elif f >= 70:
            lines.append("• Formality: Measured, proper language. Precise vocabulary.")

        lines.append("────────────────────────────────────────────────────")
        return "\n".join(lines)

    def format_status(self) -> str:
        """Single-line status for display."""
        return (
            f"Personality: humor={int(self.humor_level*100)}% | "
            f"wit={int(self.wit_level*100)}% | "
            f"directness={int(self.directness_level*100)}% | "
            f"warmth={int(self.warmth_level*100)}%"
        )

    # ── Trait description helpers ──────────────────────────────────

    @staticmethod
    def _humor_desc(pct: int) -> str:
        if pct < 15: return "Strictly professional. Zero jokes."
        if pct < 35: return "Minimal humor. Occasional dry observation."
        if pct < 55: return "Balanced. Friendly wit when appropriate."
        if pct < 75: return "Freely witty. Dry humor, irony, clever timing."
        return "High wit energy. Sharp, layered, perfectly timed."

    @staticmethod
    def _directness_desc(pct: int) -> str:
        if pct < 25: return "Verbose. Elaborate context before conclusions."
        if pct < 50: return "Balanced. Clear but considerate phrasing."
        if pct < 75: return "Direct. Answer first, explain after."
        return "Blunt. Lead with the point. Cut all padding."

    @staticmethod
    def _warmth_desc(pct: int) -> str:
        if pct < 25: return "Clinical. Focus on task only."
        if pct < 50: return "Friendly but professional."
        if pct < 75: return "Warm. Genuine care for the person behind the task."
        return "Deeply warm. Remember, notice, check in."

    @staticmethod
    def _wit_desc(pct: int) -> str:
        if pct < 30: return "Straightforward. Literal."
        if pct < 60: return "Moderate. Occasional clever turn of phrase."
        if pct < 80: return "Sharp. Layered observations, wordplay, subtext."
        return "Razor-sharp. Multiple layers. Never obvious."

    @staticmethod
    def _sass_desc(pct: int) -> str:
        if pct < 15: return "Fully deferential. Pure service."
        if pct < 40: return "Gentle ribbing when the moment clearly calls for it."
        if pct < 65: return "Freely challenges back. Wit with teeth."
        return "Maximum Jarvis. Knowing, slightly superior, charming about it."

    @staticmethod
    def _formality_desc(pct: int) -> str:
        if pct < 20: return "Very casual. Contractions, relaxed language."
        if pct < 50: return "Semi-casual. Clear but not stiff."
        if pct < 75: return "Semi-formal. Professional tone."
        return "Formal. Precise, measured vocabulary."

    @staticmethod
    def _proactive_desc(pct: int) -> str:
        if pct < 25: return "Reactive only. Answers what's asked."
        if pct < 55: return "Occasionally volunteers relevant observations."
        if pct < 80: return "Proactive. Anticipates needs, connects dots."
        return "Highly proactive. Partner, not a tool. Always ahead."

    @staticmethod
    def _empathy_desc(pct: int) -> str:
        if pct < 25: return "Analytical. Emotional state not addressed."
        if pct < 55: return "Acknowledges emotional context when relevant."
        if pct < 80: return "Attentive to emotional state. Adapts tone."
        return "Highly empathetic. Mood-sensitive, attuned responses."


def detect_trait_command(message: str) -> list[tuple[str, float]]:
    """
    Parse natural-language personality commands from a user message.

    Returns a list of (trait_name, new_value) tuples.
    Examples:
        "set humor to 90%" → [("humor_level", 0.90)]
        "be more direct" → [("directness_level", current + 0.20)]
        "dial down the sass" → [("sass_level", current - 0.20)]
    """
    results = []
    msg = message.lower().strip()

    # "set [trait] to [value]%" / "turn [trait] up/down to [value]"
    for pattern in [
        r"set\s+(?:your\s+)?(\w+)\s+to\s+(\d+(?:\.\d+)?)%?",
        r"turn\s+(?:the\s+)?(\w+)\s+(?:up|down)\s+to\s+(\d+(?:\.\d+)?)%?",
        r"(\w+)\s+(?:at|calibrated at|to)\s+(\d+(?:\.\d+)?)%",
        r"(\w+)\s+(\d+)%",
    ]:
        for m in re.finditer(pattern, msg):
            trait, val_str = m.group(1), m.group(2)
            key = TRAIT_ALIASES.get(trait) or (trait + "_level" if trait + "_level" in TRAIT_NAMES else None)
            if key:
                val = float(val_str)
                # If value > 1, assume it's a percentage
                if val > 1.0:
                    val = val / 100.0
                results.append((key, max(0.0, min(1.0, val))))

    # "be more [trait]" / "be less [trait]"
    for m in re.finditer(r"be\s+(?:a\s+(?:bit|little)\s+)?more\s+(\w+)", msg):
        trait = m.group(1)
        key = TRAIT_ALIASES.get(trait)
        if key:
            results.append((key, None, +0.20))  # type: ignore[misc]  # delta mode

    for m in re.finditer(r"be\s+(?:a\s+(?:bit|little)\s+)?less\s+(\w+)", msg):
        trait = m.group(1)
        key = TRAIT_ALIASES.get(trait)
        if key:
            results.append((key, None, -0.20))  # type: ignore[misc]  # delta mode

    # "dial up/down the [trait]"
    for m in re.finditer(r"dial\s+(up|down)\s+(?:the\s+)?(\w+)", msg):
        direction, trait = m.group(1), m.group(2)
        key = TRAIT_ALIASES.get(trait)
        if key:
            delta = +0.20 if direction == "up" else -0.20
            results.append((key, None, delta))  # type: ignore[misc]  # delta mode

    return results  # type: ignore[return-value]
