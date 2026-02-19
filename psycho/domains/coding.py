"""
Coding domain handler — Python execution, syntax highlighting, code block detection.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .base import DomainHandler, DomainResult

if TYPE_CHECKING:
    from psycho.agent.context import AgentContext
    from psycho.llm.base import LLMProvider
    from psycho.storage.database import Database


@dataclass
class ExecutionResult:
    code: str
    language: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    elapsed_ms: float = 0.0
    timed_out: bool = False
    error: str = ""

    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.error

    def display(self) -> str:
        parts = []
        if self.stdout:
            parts.append(f"Output:\n{self.stdout.strip()}")
        if self.stderr:
            parts.append(f"Stderr:\n{self.stderr.strip()[:500]}")
        if self.timed_out:
            parts.append("Execution timed out (10s limit)")
        if self.error:
            parts.append(f"Error: {self.error}")
        status = "✓" if self.success() else "✗"
        timing = f" ({self.elapsed_ms:.0f}ms)"
        return f"{status} Executed{timing}\n" + "\n".join(parts) if parts else f"{status} No output{timing}"


# Regex to find code blocks in markdown responses
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Execution triggers in user message
_RUN_TRIGGERS = re.compile(
    r"\b(run|execute|test|try|eval|check|verify|see if|can you run|please run)\b",
    re.IGNORECASE,
)

EXECUTION_TIMEOUT = 10  # seconds


class CodeExecutor:
    """
    Safely executes Python code in a subprocess with a timeout.

    Isolation: separate process, temp file, no network (by convention),
    stdout/stderr captured, hard timeout enforced.
    """

    async def execute(self, code: str, language: str = "python") -> ExecutionResult:
        if language not in ("python", "py", "python3"):
            return ExecutionResult(
                code=code,
                language=language,
                error=f"Execution only supported for Python (got '{language}')",
            )

        start = time.time()
        result = ExecutionResult(code=code, language=language)

        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir(),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=EXECUTION_TIMEOUT
                )
                result.stdout = stdout_bytes.decode("utf-8", errors="replace")[:3000]
                result.stderr = stderr_bytes.decode("utf-8", errors="replace")[:1000]
                result.exit_code = proc.returncode or 0
            except asyncio.TimeoutError:
                proc.kill()
                result.timed_out = True
                result.error = f"Timed out after {EXECUTION_TIMEOUT}s"
        except Exception as e:
            result.error = str(e)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        result.elapsed_ms = (time.time() - start) * 1000
        logger.debug(
            f"Code executed: exit={result.exit_code}, "
            f"{result.elapsed_ms:.0f}ms, "
            f"{'TIMEOUT' if result.timed_out else 'OK'}"
        )
        return result


class CodingHandler(DomainHandler):
    """
    Domain handler for coding/programming interactions.

    Features:
        - Detects code blocks in agent responses
        - Auto-executes Python when user asks "run this" / "test this"
        - Language detection from code blocks
        - Adds coding-specific system prompt instructions
    """

    def __init__(self, db: "Database", llm: "LLMProvider") -> None:
        super().__init__(db, llm)
        self._executor = CodeExecutor()

    @property
    def domain_name(self) -> str:
        return "coding"

    def system_addendum(self, ctx: "AgentContext") -> str:
        return (
            "\nFor coding questions: always include working code examples. "
            "Use proper syntax highlighting markers (```python). "
            "For Python, prefer modern idioms (f-strings, type hints, dataclasses). "
            "Be explicit about Python version when relevant (3.11+)."
        )

    async def post_process(
        self, ctx: "AgentContext", response: str
    ) -> DomainResult:
        result = DomainResult(domain="coding")

        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)
        result.code_blocks = code_blocks

        # Check if user wants execution
        wants_execution = bool(_RUN_TRIGGERS.search(ctx.user_message))
        python_blocks = [b for b in code_blocks if b["language"] in ("python", "py", "python3", "")]

        if wants_execution and python_blocks:
            # Execute the first Python block
            block = python_blocks[0]
            exec_result = await self._executor.execute(block["code"], block["language"])

            result.structured_data["execution"] = {
                "code": block["code"][:500],
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "exit_code": exec_result.exit_code,
                "elapsed_ms": exec_result.elapsed_ms,
                "success": exec_result.success(),
            }

            # Build a Rich-formatted execution panel
            status_color = "green" if exec_result.success() else "red"
            status = "PASS" if exec_result.success() else "FAIL"
            lines = [
                f"[{status_color}]── Execution Result ({exec_result.elapsed_ms:.0f}ms) ──[/{status_color}]"
            ]
            if exec_result.stdout:
                lines.append(exec_result.stdout.strip()[:800])
            if exec_result.stderr:
                lines.append(f"[red]{exec_result.stderr.strip()[:400]}[/red]")
            if exec_result.timed_out:
                lines.append(f"[red]Timed out after {EXECUTION_TIMEOUT}s[/red]")

            result.add_extra("\n".join(lines))
            result.add_action(
                f"Executed Python code: {status} ({exec_result.elapsed_ms:.0f}ms)"
            )

        elif code_blocks and not wants_execution:
            result.add_action(f"Detected {len(code_blocks)} code block(s)")

        return result

    @staticmethod
    def _extract_code_blocks(text: str) -> list[dict]:
        """Extract all code blocks from markdown text."""
        blocks = []
        for match in _CODE_BLOCK_RE.finditer(text):
            lang = match.group(1).lower().strip() or "text"
            code = match.group(2).strip()
            if code:
                blocks.append({"language": lang, "code": code})
        return blocks
