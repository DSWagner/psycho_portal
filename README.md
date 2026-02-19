# PsychoPortal

**Self-evolving AI personal assistant with a persistent knowledge graph.**

> Remembers everything. Learns from mistakes. Gets smarter every session.

---

## What It Does

PsychoPortal is a local-first AI assistant that:

- **Builds its own knowledge graph** from your conversations — entities, relationships, facts — all structured and searchable
- **Learns from mistakes** — when you correct it, it updates its confidence scores and avoids the same error in future sessions
- **Persists memory across sessions** — everything you discuss is remembered and used as context in future conversations
- **Works with any LLM** — Anthropic Claude (API) or any local model via Ollama
- **Runs entirely on your machine** — no cloud required except for the API key

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   PSYCHO PORTAL                              │
│                                                              │
│  CLI Dashboard (Rich + prompt_toolkit)                       │
│         ↓                                                    │
│  Agent Core (perceive → think → act → learn)                 │
│         ↓                      ↓                             │
│  LLM Provider             Memory Manager                     │
│  (Anthropic / Ollama)     (Short + Long + Semantic)          │
│         ↓                      ↓                             │
│                        Knowledge Graph                       │
│                        (NetworkX + ChromaDB)                 │
└─────────────────────────────────────────────────────────────┘
```

### The Self-Evolution Loop

```
Interaction
    ↓
Extract entities & relationships (LLM-powered)
    ↓
Update knowledge graph (confidence-weighted)
    ↓
Post-session reflection (synthesize, infer, prune)
    ↓
Next session is smarter
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Current | Foundation: agent, memory, CLI dashboard |
| 2 | 🔜 Next | Semantic memory (ChromaDB) + Ollama |
| 3 | 📋 Planned | Knowledge graph (NetworkX) |
| 4 | 📋 Planned | Self-evolution engine (reflection, confidence) |
| 5 | 📋 Planned | Domain intelligence (coding, health, tasks) |
| 6 | 📋 Planned | FastAPI server + web UI |

## Quick Start

### 1. Install Python 3.11+

### 2. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/psycho_portal
cd psycho_portal
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Run

```bash
# Interactive chat with dashboard
python main.py chat

# View memory statistics
python main.py stats

# Start API server (Phase 6)
python main.py serve
```

## Configuration

All configuration lives in `.env`:

```env
# LLM Provider: "anthropic" or "ollama"
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001  # cheapest/fastest

# For local (no API key needed):
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py chat` | Start interactive chat |
| `python main.py stats` | Memory & session stats |
| `python main.py serve` | HTTP API server |

### In-chat commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/stats` | Session statistics |
| `/facts` | List stored facts |
| `/clear` | Clear screen |
| `exit` / `quit` | Exit chat |

## Data

All personal data is stored in `data/` (gitignored):

```
data/
├── psycho.db           # SQLite: conversations, facts, preferences
├── graph/              # Knowledge graph (Phase 3)
├── vectors/            # ChromaDB embeddings (Phase 2)
└── journals/           # Session reflection logs (Phase 4)
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | Best AI/ML ecosystem |
| LLM (API) | Anthropic Claude Haiku | Cheapest capable model |
| LLM (local) | Ollama | Zero-cost local inference |
| Knowledge graph | NetworkX | In-process, JSON serializable, zero infrastructure |
| Vector store | ChromaDB | Local, no Docker, pluggable embeddings |
| Database | SQLite + aiosqlite | Zero setup, async, fully capable |
| CLI | Rich + Click + prompt_toolkit | Beautiful terminal, input with history |
| Config | pydantic-settings | Typed, validated, .env-backed |

## License

MIT
