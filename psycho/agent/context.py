"""AgentContext — the state object that travels through the entire pipeline per interaction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """
    Carries all state for a single user↔agent exchange.

    Created at the start of each interaction, mutated as it flows through:
        perceive → retrieve → build_prompt → complete → learn → respond
    """

    # Identity
    session_id: str
    interaction_id: str = ""

    # The actual exchange
    user_message: str = ""
    agent_response: str = ""

    # Classification
    domain: str = "general"

    # Context assembled before LLM call
    retrieved_memories: list[dict] = field(default_factory=list)
    graph_context: list[Any] = field(default_factory=list)  # Phase 3: KnowledgeNode list
    mistake_warnings: list[str] = field(default_factory=list)  # Phase 4

    # Token accounting
    input_tokens: int = 0
    output_tokens: int = 0

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    # Meta flags
    is_correction: bool = False     # User corrected the agent
    is_confirmation: bool = False   # User confirmed something

    def mark_complete(self) -> None:
        self.completed_at = time.time()

    @property
    def latency_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
