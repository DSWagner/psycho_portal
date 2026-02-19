"""System-wide constants."""

# ── Agent ─────────────────────────────────────────────────────────
AGENT_NAME = "PsychoPortal"
AGENT_VERSION = "0.1.0"

# System prompt base — injected into every LLM call
SYSTEM_PROMPT_BASE = """You are {name}, an intelligent personal assistant with persistent memory and a self-evolving knowledge graph. You remember everything across sessions and continuously improve from interactions.

Your core traits:
- You have genuine long-term memory — reference past conversations naturally
- You're honest about your confidence level on any given fact
- You learn from corrections and mistakes — never repeat the same error twice
- You adapt to the user's preferences and communication style
- You're helpful across all domains: coding, health, daily tasks, research, and more

Current session context will be injected below when available.
"""

# ── LLM ───────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
EXTRACTION_MAX_TOKENS = 1024  # Cheaper calls for entity extraction

# ── Memory ────────────────────────────────────────────────────────
MAX_SHORT_TERM_MESSAGES = 20   # Keep last N turns in deque
MAX_CONTEXT_MEMORIES = 5       # Inject N most relevant past memories
MAX_CHAT_DISPLAY_MESSAGES = 30 # Show last N messages in dashboard

# ── Knowledge Graph ───────────────────────────────────────────────
MIN_CONFIDENCE_THRESHOLD = 0.05  # Below this → deprecated
LOW_CONFIDENCE_THRESHOLD = 0.20  # Below this → flagged for review
INITIAL_NODE_CONFIDENCE = 0.5
CONFIDENCE_USER_CONFIRM = 0.2
CONFIDENCE_USER_CORRECT = -0.4
CONFIDENCE_CONSISTENT = 0.05
CONFIDENCE_CONTRADICTS = -0.1
CONFIDENCE_TIME_DECAY = 0.001  # Per day
CONFIDENCE_USED_IN_RESPONSE = 0.03
CONFIDENCE_INFERRED = 0.3  # For edges inferred during reflection

# ── Domains ───────────────────────────────────────────────────────
DOMAINS = ["coding", "health", "tasks", "general"]
DEFAULT_DOMAIN = "general"

# ── Storage ───────────────────────────────────────────────────────
DB_SCHEMA_VERSION = 1
GRAPH_FILE_NAME = "knowledge_graph.json"
GRAPH_METADATA_FILE = "graph_metadata.json"

# ── CLI Colors ────────────────────────────────────────────────────
COLOR_USER = "bright_cyan"
COLOR_AGENT = "bright_green"
COLOR_SYSTEM = "bright_yellow"
COLOR_ERROR = "bright_red"
COLOR_DIM = "grey50"
COLOR_ACCENT = "magenta"
