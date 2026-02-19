"""Short-term memory — in-process conversation buffer, never persisted."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from psycho.llm.base import Message


@dataclass
class Turn:
    """A single exchange in the conversation."""

    user: str
    assistant: str


class ShortTermMemory:
    """
    Fixed-size deque of recent conversation turns.

    Lives only in memory for the duration of the process.
    Acts as the immediate context window for the LLM.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, user_message: str, assistant_response: str) -> None:
        self._turns.append(Turn(user=user_message, assistant=assistant_response))

    def get_messages(self) -> list[Message]:
        """Flatten turns into LLM Message list (alternating user/assistant)."""
        messages = []
        for turn in self._turns:
            messages.append(Message(role="user", content=turn.user))
            messages.append(Message(role="assistant", content=turn.assistant))
        return messages

    def get_turns(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def is_empty(self) -> bool:
        return len(self._turns) == 0
