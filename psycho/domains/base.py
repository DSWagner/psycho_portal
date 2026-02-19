"""Domain handler base class and shared data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from psycho.agent.context import AgentContext
    from psycho.llm.base import LLMProvider
    from psycho.storage.database import Database


@dataclass
class DomainResult:
    """
    Output of a domain handler's post-processing pass.

    After the LLM generates a response, the domain handler may:
        - Extract structured data (tasks, health metrics, code blocks)
        - Enrich the display (syntax highlighting, metric tables)
        - Trigger side effects (log metric, create task, run code)
        - Suggest follow-up actions to the user
    """

    domain: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    display_extras: list[str] = field(default_factory=list)   # Rich-formatted addons
    actions_taken: list[str] = field(default_factory=list)    # What was auto-done
    suggestions: list[str] = field(default_factory=list)      # Suggestions for user
    code_blocks: list[dict] = field(default_factory=list)     # Extracted code blocks
    is_empty: bool = True

    def add_action(self, action: str) -> None:
        self.actions_taken.append(action)
        self.is_empty = False

    def add_extra(self, extra: str) -> None:
        self.display_extras.append(extra)
        self.is_empty = False


class DomainHandler(ABC):
    """
    Abstract domain handler.

    Each domain handler provides:
        system_addendum()   — extra text injected into the system prompt
        post_process()      — runs after LLM response, extracts/logs structured data
        format_display()    — optional Rich-formatted extra content to show after response
    """

    def __init__(self, db: "Database", llm: "LLMProvider") -> None:
        self._db = db
        self._llm = llm

    @property
    @abstractmethod
    def domain_name(self) -> str:
        ...

    def system_addendum(self, ctx: "AgentContext") -> str:
        """
        Extra instructions injected into the system prompt for this domain.
        Keep short (< 200 chars). Return empty string if not needed.
        """
        return ""

    @abstractmethod
    async def post_process(
        self, ctx: "AgentContext", response: str
    ) -> DomainResult:
        """
        Extract structured data and trigger side effects after LLM response.
        Must be fast — runs before showing the response to the user.
        """
        ...
