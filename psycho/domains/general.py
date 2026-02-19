"""General domain handler — fallback for all non-specialized queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DomainHandler, DomainResult

if TYPE_CHECKING:
    from psycho.agent.context import AgentContext
    from psycho.llm.base import LLMProvider
    from psycho.storage.database import Database


class GeneralHandler(DomainHandler):
    """Fallback handler — passes through with no special processing."""

    @property
    def domain_name(self) -> str:
        return "general"

    def system_addendum(self, ctx: "AgentContext") -> str:
        return ""

    async def post_process(
        self, ctx: "AgentContext", response: str
    ) -> DomainResult:
        return DomainResult(domain="general")
