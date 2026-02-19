"""
Session journal — writes a human-readable and machine-readable record
of each session's learnings, corrections, and quality metrics.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger


class SessionJournal:
    """
    Writes JSON + Markdown session journals to data/journals/.

    Each journal captures:
        - Session metadata (id, duration, message count)
        - LLM-synthesized summary
        - Key learnings extracted
        - Corrections detected
        - Behavioral patterns observed
        - Knowledge gaps identified
        - Quality score
        - Graph evolution stats
        - Insights generated

    Files written:
        data/journals/YYYY-MM-DD_<session_id>.json    — machine-readable
        data/journals/YYYY-MM-DD_<session_id>.md      — human-readable
    """

    def __init__(self, journal_dir: Path) -> None:
        self._dir = journal_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        session_id: str,
        started_at: float,
        reflection_data: dict,
        graph_changes: dict,
        message_count: int,
    ) -> Path:
        """
        Write the session journal and return the JSON file path.

        reflection_data: output from the ReflectionEngine (LLM synthesis)
        graph_changes: stats dict from GraphEvolver.integrate()
        """
        ended_at = time.time()
        duration_min = (ended_at - started_at) / 60

        date_str = datetime.now().strftime("%Y-%m-%d")
        stem = f"{date_str}_{session_id}"
        json_path = self._dir / f"{stem}.json"
        md_path = self._dir / f"{stem}.md"

        # Build journal document
        journal = {
            "session_id": session_id,
            "date": date_str,
            "started_at": datetime.fromtimestamp(started_at).strftime("%H:%M:%S"),
            "ended_at": datetime.fromtimestamp(ended_at).strftime("%H:%M:%S"),
            "duration_minutes": round(duration_min, 1),
            "message_count": message_count,
            "quality_score": reflection_data.get("quality_score", 0.0),
            "summary": reflection_data.get("session_summary", ""),
            "key_learnings": reflection_data.get("key_learnings", []),
            "corrections_detected": reflection_data.get("corrections_detected", []),
            "patterns_observed": reflection_data.get("patterns_observed", []),
            "knowledge_gaps": reflection_data.get("knowledge_gaps", []),
            "insights": reflection_data.get("insights", []),
            "graph_changes": graph_changes,
        }

        # Write JSON
        json_path.write_text(
            json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Write Markdown
        md_path.write_text(self._to_markdown(journal), encoding="utf-8")

        logger.info(f"Session journal written: {json_path}")
        return json_path

    @staticmethod
    def _to_markdown(j: dict) -> str:
        quality = j.get("quality_score", 0.0)
        quality_emoji = "🟢" if quality > 0.75 else "🟡" if quality > 0.5 else "🔴"
        bars = "█" * round(quality * 10) + "░" * (10 - round(quality * 10))

        lines = [
            f"# Session Journal — {j['date']} @ {j['started_at']}",
            "",
            "## Overview",
            f"| | |",
            f"|---|---|",
            f"| **Session ID** | `{j['session_id']}` |",
            f"| **Duration** | {j['duration_minutes']} minutes |",
            f"| **Messages** | {j['message_count']} |",
            f"| **Quality** | {quality_emoji} {bars} {quality:.2f} |",
            "",
        ]

        if j.get("summary"):
            lines += ["## Summary", j["summary"], ""]

        if j.get("key_learnings"):
            lines.append("## Key Learnings")
            for item in j["key_learnings"]:
                fact = item.get("fact", item) if isinstance(item, dict) else item
                domain = item.get("domain", "") if isinstance(item, dict) else ""
                conf = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5
                d_str = f" `[{domain}]`" if domain else ""
                lines.append(f"- {fact}{d_str} _(confidence: {conf:.2f})_")
            lines.append("")

        if j.get("corrections_detected"):
            lines.append("## Corrections Detected")
            for c in j["corrections_detected"]:
                wrong = c.get("wrong", "?")
                correct = c.get("correct", "?")
                lines.append(f"- ~~{wrong}~~ → **{correct}**")
            lines.append("")

        if j.get("insights"):
            lines.append("## Insights")
            for ins in j["insights"]:
                text = ins.get("insight", ins) if isinstance(ins, dict) else ins
                basis = ins.get("basis", "") if isinstance(ins, dict) else ""
                lines.append(f"- {text}")
                if basis:
                    lines.append(f"  _Based on: {basis}_")
            lines.append("")

        if j.get("patterns_observed"):
            lines.append("## Patterns Observed")
            for p in j["patterns_observed"]:
                pattern = p.get("pattern", p) if isinstance(p, dict) else p
                impl = p.get("implication", "") if isinstance(p, dict) else ""
                lines.append(f"- {pattern}")
                if impl:
                    lines.append(f"  → _{impl}_")
            lines.append("")

        if j.get("knowledge_gaps"):
            lines.append("## Knowledge Gaps")
            for g in j["knowledge_gaps"]:
                topic = g.get("topic", g) if isinstance(g, dict) else g
                why = g.get("why_insufficient", "") if isinstance(g, dict) else ""
                lines.append(f"- **{topic}**: {why}")
            lines.append("")

        gc = j.get("graph_changes", {})
        if any(gc.values()):
            lines += [
                "## Graph Evolution",
                f"| Nodes added | {gc.get('nodes_added', 0)} |",
                f"| Edges added | {gc.get('edges_added', 0)} |",
                f"| Facts added | {gc.get('facts_added', 0)} |",
                f"| Corrections | {gc.get('corrections_applied', 0)} |",
                "",
            ]

        lines.append("---")
        lines.append(f"_Generated by PsychoPortal · {j['date']}_")
        return "\n".join(lines)

    def load_latest(self, n: int = 5) -> list[dict]:
        """Load the N most recent journal files."""
        files = sorted(self._dir.glob("*.json"), reverse=True)[:n]
        journals = []
        for f in files:
            try:
                journals.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return journals
