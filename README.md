# PsychoPortal

**A self-evolving AI personal assistant with a persistent knowledge graph.**

> Remembers everything. Learns from every conversation. Gets smarter every session.
> Talk to it by voice, upload files, and watch it build a living knowledge graph in real time.
> Works with Claude API (dev) or any local model via Ollama.

---

## What It Does

PsychoPortal is not a wrapper around an LLM. It is an autonomous learning system that:

- **Builds a knowledge graph** from your conversations — entities, relationships, facts, preferences — all structured, confidence-weighted, and growing
- **Learns from mistakes** — when you correct it, it drops confidence on the wrong belief, indexes the correction semantically, and injects a warning before similar future questions
- **Reflects after sessions** — synthesizes what was learned, derives insights, writes a journal, updates the graph, and runs maintenance
- **Remembers everything** across sessions — semantic vector search (ChromaDB) finds relevant past conversations by meaning, not just keywords
- **Ingests any file** — `.py`, `.md`, `.pdf`, `.json`, `.yaml`, `.csv`, images — all parsed and absorbed into the knowledge graph
- **Understands images** — upload screenshots, diagrams, charts, handwritten notes, or code screenshots; Claude Vision extracts every piece of knowledge from them
- **Voice mode** — full duplex voice conversation: speak to it, it speaks back, with an animated morphing blob that reacts to audio in real time
- **Tracks your health** — mentions of weight, sleep, calories, etc. are auto-logged silently
- **Manages your tasks** — "remind me to X" creates tasks with inferred priority and due date
- **Runs Python code** — ask "run this" and it executes safely in a sandboxed subprocess
- **Session history** — browse all past conversations in the web UI

---

## Architecture

```
User Input (text or voice)
    │
    ▼
Signal Detector ──── correction/confirmation → real-time confidence update
    │
Domain Router ─────── coding / health / tasks / general
    │
    ├─ Semantic Memory (ChromaDB) ─── finds relevant past conversations
    ├─ Knowledge Graph (NetworkX) ─── retrieves relevant nodes by meaning + PageRank
    └─ Mistake Warnings ───────────── injects "known failure patterns" from past errors
    │
    ▼
LLM (Claude Haiku / Ollama)
    │
    ▼
Domain Handler ─────── code execution / metric logging / task creation
    │
    ▼
Response to user (text + optional TTS → voice)
    │ (background)
    ├─ Extract entities/relations/facts → Knowledge Graph
    ├─ Store in ChromaDB (semantic memory)
    ├─ Log to episodic event log
    └─ Record mistake (if correction was detected)

── On session exit ──────────────────────────────────────────────
Post-Session Reflection:
    LLM synthesizes session → quality score, learnings, corrections
    → update graph confidence → derive insights → run graph maintenance
    → write session journal (JSON + Markdown) → save everything
```

### The Four Memory Systems

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Short-term** | In-process deque | Last 20 turns, immediate LLM context |
| **Long-term** | SQLite | All interactions, facts, preferences |
| **Semantic** | ChromaDB (ONNX embeddings) | Find relevant past conversations by meaning |
| **Episodic** | SQLite event log | Ordered timeline of what happened when |

### Knowledge Graph

- **12 node types**: concept, entity, person, technology, fact, preference, skill, mistake, question, topic, file, event
- **16 edge types**: is_a, part_of, relates_to, contradicts, corrects, preferred_by, extracted_from, inferred_from…
- Every node has a **confidence score** (0.0–1.0) updated by: user corrections (−0.4), confirmations (+0.2), time decay (−0.001/day), reinforcement (+0.03)
- Nodes below 0.05 confidence are deprecated (kept as history, not used in responses)
- **PageRank** computes node importance — frequently-connected nodes appear in more contexts

---

## Requirements

- **Python 3.11+**
- **Anthropic API key** — get one free at [console.anthropic.com](https://console.anthropic.com) (Claude Haiku is very cheap, ~$0.001 per conversation)
- **OR** [Ollama](https://ollama.com) installed locally with a model pulled (e.g., `ollama pull llama3.2`)
- ~500MB disk for the ONNX embedding model (downloaded once automatically on first run)
- For voice mode: Chrome or Edge (Web Speech API required for STT)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DSWagner/psycho_portal
cd psycho_portal
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows (Command Prompt):
venv\Scripts\activate

# Windows (Git Bash / PowerShell):
./venv/Scripts/activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First install downloads ~500MB (ChromaDB ONNX model + torch). Subsequent runs are instant.

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` in any text editor and set your key:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_REAL_KEY_HERE
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

**Get your API key:** [console.anthropic.com → API Keys → Create Key](https://console.anthropic.com)

> To use a local model instead (no API key needed):
> ```env
> LLM_PROVIDER=ollama
> OLLAMA_MODEL=llama3.2
> OLLAMA_BASE_URL=http://localhost:11434
> ```
> Then run `ollama serve` and `ollama pull llama3.2` first.

---

## Quick Start

```bash
# Start the chat dashboard (CLI)
python main.py chat

# Start the web UI
python main.py serve
# → open http://localhost:8000
```

On first launch:
1. The ONNX embedding model is downloaded (~79MB, one time only)
2. A SQLite database is created at `data/psycho.db`
3. Your knowledge graph starts empty and grows from conversations

---

## Running the Application

### Web UI (recommended)

```bash
python main.py serve
```

Opens the API server at **http://localhost:8000** — visit in your browser for the full web UI.

**Web UI features:**
- Streaming chat with markdown rendering and code highlighting
- **Session history panel** (left sidebar) — browse and replay all past conversations
- **New Chat button** — clears the display while preserving all memory
- **File upload** (📎 button or drag & drop) — ingest any file directly from your browser
- **Voice mode** (🎤 Voice button) — full voice call experience with animated blob
- Live D3.js knowledge graph visualization
- Stats panel, task list, text ingest

### Chat (CLI Dashboard)

```bash
python main.py chat
```

The dashboard shows a Rich terminal UI with your conversation, typing history (arrow keys), and inline stats.

### Statistics

```bash
python main.py stats
```

Shows session history, graph stats, memory counts.

### Knowledge Graph Inspector

```bash
# Show top 25 nodes
python main.py graph

# Filter by type
python main.py graph --type preference

# Export to D3.js JSON for custom visualization
python main.py graph --export graph.json
```

### File Ingestion

```bash
# Single file
python main.py ingest notes.md
python main.py ingest report.pdf
python main.py ingest mycode.py
python main.py ingest screenshot.png   # vision-extracted via Claude

# Entire folder (recursive)
python main.py ingest ./docs/
python main.py ingest ./src/

# Raw text
python main.py ingest "Python uses indentation for blocks" --text
```

**Supported file types:** `.txt`, `.md`, `.rst`, `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.cpp`, `.c`, `.json`, `.yaml`, `.toml`, `.csv`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`

> **Images** are processed using Claude's vision API — every word, chart, diagram, code screenshot, and UI element is extracted and stored in the knowledge graph. Requires Anthropic provider.

### Task Manager

```bash
# View pending tasks
python main.py tasks

# Add a task
python main.py tasks --add "Review PR #47" --priority high

# Complete a task (by ID prefix or title match)
python main.py tasks --done "Review PR"

# View completed tasks
python main.py tasks --all
```

### Health Dashboard

```bash
python main.py health           # Last 30 days
python main.py health --days 7  # Last week
```

### Post-Session Reflection

```bash
python main.py reflect
```

Manually triggers the reflection engine on your most recent session data.

### Reset (start fresh)

```bash
python main.py reset        # Asks for confirmation
python main.py reset --yes  # Wipes everything immediately
```

Deletes all memory: knowledge graph, conversation history, vectors, journals, health and tasks data. Your `.env` config is untouched.

---

## Voice Mode

Voice mode gives you a real-time voice conversation experience directly in the browser.

### How to use

1. Run `python main.py serve` and open `http://localhost:8000`
2. Click **🎤 Voice** in the top-right header
3. Click the **microphone button** (or press `Space`) to start speaking
4. Speak naturally — the transcript appears in real time
5. The agent processes your message and speaks the response
6. The mic reopens automatically for continuous conversation
7. Press `Esc` or click **✕ End Call** to exit

### The Animated Blob

The central blob reacts to the current state:

| State | Color | Animation |
|-------|-------|-----------|
| **Ready** | Purple | Slow gentle breathing |
| **Listening** | Cyan | Faster pulse + expanding rings |
| **Thinking** | Amber | Fast complex morphing |
| **Speaking** | Pink/Magenta | Intense warping driven by audio amplitude |

### Microphone selection

Voice mode uses the browser's **Web Speech API**, which routes audio through your OS default recording device. To change which microphone is used:
- **Windows**: Right-click the speaker icon → Sound settings → Input → choose device
- **Chrome**: `chrome://settings/content/microphone` → select device per-site
- **Edge**: `edge://settings/content/microphone` → select device per-site

> Voice input requires Chrome or Edge. Firefox does not support the Web Speech API. The agent will show a warning if the browser is unsupported.

### TTS Provider options

Configure in `.env` (cheapest first):

| Provider | Quality | Cost | Config |
|----------|---------|------|--------|
| `browser` (default) | Good | **Free** | None |
| `openai` | High | ~$0.015/1k chars | `OPENAI_API_KEY` + `TTS_VOICE` |
| `elevenlabs` | Highest | Paid | `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID` |

```env
# Default — no config needed, uses browser Web Speech API
TTS_PROVIDER=browser

# OpenAI TTS
TTS_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
TTS_VOICE=alloy   # alloy | echo | fable | onyx | nova | shimmer

# ElevenLabs
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your-key-here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel (default)
```

---

## In-Chat Commands

While in `python main.py chat`, type these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/stats` | Memory, graph, and session statistics |
| `/graph` | Inspect the top 20 knowledge graph nodes |
| `/tasks` | View pending tasks |
| `/health` | View logged health metrics |
| `/facts` | List stored facts with confidence scores |
| `/ingest <path>` | Ingest a file or folder into the graph |
| `/reflect` | Run post-session reflection right now |
| `/mistakes` | Show all recorded past mistakes |
| `/clear` | Clear the screen |
| `exit` / `quit` | Exit (triggers reflection automatically) |

---

## How the Agent Learns

### During a conversation

1. **Signal detection** — Before every response, the agent checks if you're correcting or confirming it. "No that's wrong" immediately drops confidence on the relevant graph node.

2. **Graph context injection** — Relevant knowledge graph nodes are retrieved via semantic search + graph traversal and injected into the system prompt before every LLM call.

3. **Mistake warnings** — If you've corrected the agent on a similar question before, that warning appears in the prompt: *"Previously, when asked X, you said something incorrect. The correct answer is Y."*

4. **Background extraction** — After every response, a cheap LLM call extracts entities, relationships, facts, and corrections from the exchange and adds them to the graph.

5. **Agent identity** — If you give the agent a name (e.g. "your name is Raz"), it stores that as a high-confidence preference node and uses it permanently across all sessions.

### At session end (exit or `/reflect`)

The **Reflection Engine** runs a full pipeline:

1. Reviews the last 25 interactions
2. LLM synthesizes: quality score, key learnings, corrections found, patterns, knowledge gaps
3. Updates graph confidence (boost confirmed beliefs, drop incorrect ones)
4. Adds key learnings as `FACT` nodes
5. Records corrections as indexed mistakes (searchable next session)
6. Derives insights by combining multiple graph nodes
7. Runs graph maintenance: deduplication, pruning, time decay, PageRank
8. Writes a session journal to `data/journals/YYYY-MM-DD_sessionID.json` (+ `.md`)
9. Saves the full knowledge graph to disk

---

## Web UI API Reference

When running `python main.py serve`, the following endpoints are available.
Interactive docs at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI (index.html) |
| `GET` | `/api/stats` | Agent statistics |
| `POST` | `/api/chat` | Send message, get response |
| `GET` | `/api/history` | Recent chat history |
| `POST` | `/api/ingest` | Ingest text into graph |
| `POST` | `/api/upload` | Upload and ingest a file (multipart) |
| `GET` | `/api/sessions` | List all past sessions |
| `GET` | `/api/sessions/{id}/messages` | Get messages for a specific session |
| `GET` | `/api/graph` | Knowledge graph (D3.js format) |
| `GET` | `/api/tasks` | List tasks |
| `POST` | `/api/tasks` | Create task |
| `PATCH` | `/api/tasks/{id}/complete` | Complete task |
| `GET` | `/api/health-metrics` | Health metric summary |
| `POST` | `/api/health-metrics` | Log a health metric |
| `GET` | `/api/voice/config` | Active TTS provider info |
| `POST` | `/api/voice/tts` | Text-to-speech (returns MP3) |
| `WS` | `/ws/chat` | Streaming WebSocket chat |
| `GET` | `/docs` | Interactive API docs (Swagger) |

### WebSocket Protocol

```json
// Send:
{ "type": "chat", "message": "your message" }

// Receive (streaming tokens):
{ "type": "token", "token": "Hello" }
// … repeated for each token

// Final message:
{ "type": "done", "response": "full response", "domain": "coding", "actions": ["Task created: Buy milk"] }

// On error:
{ "type": "error", "message": "description" }
```

---

## Configuration Reference (`.env`)

```env
# ── LLM Provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER=anthropic           # "anthropic" or "ollama"
ANTHROPIC_API_KEY=sk-ant-...     # Your key from console.anthropic.com
ANTHROPIC_MODEL=claude-haiku-4-5-20251001  # cheapest; swap for sonnet/opus

# ── Local Model (Ollama) ──────────────────────────────────────────────────────
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434

# ── TTS Voice Mode ────────────────────────────────────────────────────────────
# TTS_PROVIDER=browser          # browser (free) | openai | elevenlabs
# OPENAI_API_KEY=sk-...         # For OpenAI TTS
# TTS_VOICE=alloy               # alloy | echo | fable | onyx | nova | shimmer
# ELEVENLABS_API_KEY=...        # For ElevenLabs TTS
# ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # ElevenLabs voice

# ── Storage (optional overrides) ──────────────────────────────────────────────
# DATA_DIR=data
# DB_PATH=data/psycho.db

# ── Agent Behavior ────────────────────────────────────────────────────────────
# REFLECTION_ENABLED=true
# EXTRACTION_ENABLED=true
# MAX_SHORT_TERM_MESSAGES=20

# ── Web Server ────────────────────────────────────────────────────────────────
# API_HOST=0.0.0.0
# API_PORT=8000
```

---

## File Structure

```
psycho_portal/
├── .env                          ← your API key (never committed)
├── .env.example                  ← copy this to .env
├── requirements.txt
├── main.py                       ← entry point
│
├── psycho/
│   ├── agent/                    ← orchestration (core, loop, context, reflection)
│   ├── llm/                      ← LLM abstraction (Anthropic + Ollama)
│   ├── memory/                   ← 4-tier memory (short, long, semantic, episodic)
│   ├── knowledge/                ← graph engine, extractor, evolver, reasoner, ingestion
│   ├── learning/                 ← mistake tracker, signal detector, journal, insights
│   ├── domains/                  ← coding, health, tasks, general + router
│   ├── storage/                  ← SQLite, ChromaDB, graph JSON store
│   ├── cli/                      ← Rich TUI, chat view, dashboard
│   └── api/                      ← FastAPI server, WebSocket, web UI
│       ├── routes/
│       │   ├── chat.py           ← chat, history, sessions, file upload
│       │   ├── graph.py          ← knowledge graph endpoints
│       │   ├── tasks.py          ← task management
│       │   ├── health_metrics.py ← health tracking
│       │   └── voice.py          ← TTS config + audio synthesis
│       └── static/
│           └── index.html        ← single-page web UI (chat + voice + history)
│
└── data/                         ← all personal data (gitignored)
    ├── psycho.db                 ← SQLite: interactions, facts, tasks, health
    ├── graph/
    │   └── knowledge_graph.json  ← your knowledge graph
    ├── vectors/                  ← ChromaDB embeddings
    ├── journals/                 ← session journals (JSON + Markdown)
    └── logs/
        └── psycho.log
```

---

## Model Cost Reference

Using Claude Haiku for development (the cheapest capable model):

| Operation | Tokens | Cost |
|-----------|--------|------|
| Single chat response | ~500–2000 | ~$0.001 |
| Knowledge extraction (background) | ~300 | ~$0.0003 |
| Post-session reflection | ~3000 | ~$0.003 |
| Domain classification | ~50 | ~$0.00005 |
| Image vision extraction | ~500–1500 | ~$0.001–0.003 |

A full day of heavy use costs roughly **$0.05–0.20**. For production, swap to `claude-sonnet-4-6` in `.env`.

---

## Troubleshooting

**"invalid x-api-key" error**
→ Your `.env` has the placeholder key. Set `ANTHROPIC_API_KEY` to your real key from [console.anthropic.com](https://console.anthropic.com).

**Slow first startup**
→ ChromaDB downloads the ONNX embedding model (~79MB) on first use. This only happens once.

**"Port 8000 already in use"**
→ Change the port: `python main.py serve --port 8001` or set `API_PORT=8001` in `.env`.

**Knowledge graph not growing**
→ Check `data/logs/psycho.log` for extraction errors. Make sure `EXTRACTION_ENABLED=true` in `.env`.

**Ollama connection refused**
→ Run `ollama serve` in a separate terminal first, then `ollama pull llama3.2`.

**Voice mode: mic not working**
→ Browser must have microphone permission. In Chrome: click the lock icon in the address bar → allow microphone. Voice input requires Chrome or Edge (Firefox lacks Web Speech API support).

**Voice mode: no sound output**
→ Default uses browser TTS (free). If you configured `TTS_PROVIDER=openai` or `elevenlabs`, check the API key is valid and the provider is reachable.

**Image ingestion returns 0 nodes**
→ Image extraction requires the Anthropic provider (`LLM_PROVIDER=anthropic`). Ollama does not support vision. Check `data/logs/psycho.log` for errors.

---

## Phase Roadmap

| Phase | Status | Feature |
|-------|--------|---------|
| 1 | ✅ Done | Foundation: agent core, 4-tier memory, Rich CLI |
| 2 | ✅ Done | Semantic memory (ChromaDB + ONNX embeddings) |
| 3 | ✅ Done | Knowledge graph + file ingestion (PDF/py/md/json/…) |
| 4 | ✅ Done | Self-evolution: reflection, mistake tracker, insights |
| 5 | ✅ Done | Domain intelligence: coding execution, health, tasks |
| 6 | ✅ Done | FastAPI server + streaming WebSocket + web UI |
| 7 | ✅ Done | Web UI v2: session history, file upload, drag & drop, image vision |
| 8 | ✅ Done | Voice mode: STT + TTS + animated blob UI |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | Best AI ecosystem |
| LLM (API) | Claude Haiku 4.5 | Cheapest capable model, ~$0.001/chat |
| LLM (local) | Ollama | Zero-cost local inference |
| Vision | Claude Vision API | Image knowledge extraction |
| Knowledge graph | NetworkX | In-process, JSON serializable, zero infra |
| Vector store | ChromaDB | Local, no Docker, ONNX embeddings |
| Embeddings | all-MiniLM-L6-v2 (ONNX) | 22MB, CPU-ready, 384-dim |
| Database | SQLite + aiosqlite | Zero setup, async, fully capable |
| CLI | Rich + Click + prompt_toolkit | Beautiful terminal, history |
| Web API | FastAPI + uvicorn | Async, streaming WebSocket |
| Web UI | Vanilla JS + D3.js + marked.js | No framework bloat, instant load |
| Voice STT | Web Speech API | Free, browser-native, zero latency |
| Voice TTS | Browser SpeechSynthesis / OpenAI / ElevenLabs | Tiered quality/cost |
| Config | pydantic-settings | Typed, validated, .env-backed |

---

## License

MIT — do whatever you want with it.

---

*Built with Claude Sonnet 4.6*
