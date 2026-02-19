#!/usr/bin/env python3
"""
PsychoPortal — Self-Evolving AI Personal Assistant
Entry point for all commands.

Usage:
    python main.py chat       # Interactive dashboard chat
    python main.py stats      # Memory & session statistics
    python main.py serve      # FastAPI HTTP server
    python main.py --help     # Full command list
"""

import sys
from pathlib import Path

# Ensure project root is in path when running as script
sys.path.insert(0, str(Path(__file__).parent))

# Configure loguru before any other imports
from pathlib import Path as _Path
from loguru import logger

_log_dir = _Path("data/logs")
_log_dir.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    _log_dir / "psycho.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
    enqueue=True,
)
# Only show WARNING+ in terminal (console output is handled by Rich)
logger.add(sys.stderr, level="WARNING", format="{level}: {message}")

from psycho.cli.app import cli

if __name__ == "__main__":
    cli()
