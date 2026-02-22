# PsychoPortal

**A self-evolving AI companion with persistent memory, TARS/Jarvis personality, and proactive intelligence.**

> Remembers everything. Learns your personality. Gets smarter every session.
> Checks in on you. Reminds you of what matters. Talks like Jarvis — wit, warmth, precision.
> Works with Claude API (dev) or any local model via Ollama. Fully switchable to offline.

---

## What It Does

PsychoPortal is not a wrapper around an LLM. It is an autonomous learning system that:

- **Has a real personality** — TARS-style adjustable traits (humor, directness, warmth, wit, sass). Set humor to 90% and watch the difference.
- **Learns YOU** — your humor style, communication preferences, thinking patterns, hobbies, interests, pet peeves. Adapts to you over time.
- **Checks in on you** — morning/evening greetings, proactive reminders, calendar alerts. Notices if you seem stressed.
- **Manages reminders** — "remind me to call mom tomorrow at 3pm" → done. Recurring, snooze, priority.
- **Calendar integration** — local calendar with optional Google Calendar sync.
- **Builds a knowledge graph** from your conversations — entities, relationships, facts, preferences — all structured, confidence-weighted, and growing.
- **Learns from mistakes** — when you correct it, it drops confidence on the wrong belief and warns before repeating.
- **Reflects after sessions** — synthesizes learnings, updates the graph, writes a journal.
- **Remembers everything** across sessions via semantic vector search.
- **Ingests any file** — `.py`, `.md`, `.pdf`, `.json`, images — parsed and absorbed into the knowledge graph.
- **Voice mode** — full duplex: speak to it, it speaks back. Animated blob reacts to audio in real time.
- **Web search** — auto-detects queries needing live data and injects results before responding.
- **Image chat** — paste any image; Claude Vision analyses it inline.
- **Interactive graph explorer** — full-screen D3 with filters, confidence slider, node detail, deletion.
- **Proactive notifications** — browser notifications for due reminders and upcoming calendar events.

---

## The Personality System (TARS/Jarvis-style)

Every trait is adjustable from 0% to 100%, just like TARS:

| Trait | Default | 0% | 100% |
|-------|---------|-----|------|
| **Humor** | 75% | Deadpan serious | Full comedian |
| **Wit** | 82% | Literal/simple | Razor-sharp layered wit |
| **Directness** | 88% | Verbose, diplomatic | Blunt, no padding |
| **Warmth** | 72% | Cold/clinical | Deeply warm |
| **Sass** | 38% | Fully deferential | Maximum Jarvis |
| **Formality** | 12% | Casual/chill | Formal/proper |
| **Proactive** | 82% | Reactive only | Always ahead |
| **Empathy** | 78% | Purely analytical | Mood-sensitive |

**Ways to adjust personality:**
- Web UI: Click **⚙ Personality** button → drag sliders
- Chat: `"set humor to 90%"` / `"be more direct"` / `"dial down the sass"`
- REST: `PATCH /api/personality { "humor_level": 0.9 }`
- `.env`: `PERSONALITY_HUMOR=0.90`

---

## Architecture

```
User Input (text / voice / image)
    │
    ├─ Personality Adapter ── TARS-style trait system + user personality learning
    │
    ▼
Signal Detector ──── correction/confirmation → real-time confidence update
    │
Domain Router ─────── coding / health / tasks / general
    │
    ├─ Semantic Memory (ChromaDB) ─── finds relevant past conversations
    ├─ Knowledge Graph (NetworkX) ─── retrieves relevant nodes by meaning + PageRank
    ├─ Reminder/Calendar context ──── injects due/upcoming events
    ├─ Check-in Engine ───────────── proactive morning/evening/return context
    └─ Mistake Warnings ───────────── injects "known failure patterns"
    │
    ▼
LLM (Claude / Ollama) — with personality-calibrated system prompt
    │
    ▼
Domain Handler ─────── code execution / metric logging / task creation / reminder creation
    │
    ▼
Response to user (text + optional TTS → voice)
    │ (background)
    ├─ Extract entities/relations/personality signals → Knowledge Graph
    ├─ Store in ChromaDB (semantic memory)
    ├─ Log to episodic event log
    └─ Record mistake (if correction was detected)

── On session exit ──────────────────────────────────────────────
Post-Session Reflection:
    LLM synthesizes session → quality score, learnings, corrections
    → update graph confidence → derive insights → run graph maintenance
    → write session journal → save personality state

── Background (ProactiveScheduler — every 60s) ─────────────────
    Check due reminders → emit notifications
    Check calendar events → emit pre-event alerts
```

### The Four Memory Systems

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Short-term** | In-process deque | Last 20 turns, immediate LLM context |
| **Long-term** | SQLite | All interactions, facts, preferences, reminders, calendar |
| **Semantic** | ChromaDB (ONNX embeddings) | Find relevant past conversations by meaning |
| **Episodic** | SQLite event log | Ordered timeline of what happened when |

### Knowledge Graph

- **12 node types**: concept, entity, person, technology, fact, preference, skill, mistake, question, topic, file, event
- **Personality nodes**: `humor_style:dry`, `interest:machine-learning`, `hobby:cycling`, `comm_style:brief`
- Every node has a **confidence score** (0.0–1.0) updated by: user corrections (−0.4), confirmations (+0.2), time decay, reinforcement
- **PageRank** computes node importance

---

## Requirements

- **Python 3.11+**
- **Anthropic API key** — get one free at [console.anthropic.com](https://console.anthropic.com)
- **OR** [Ollama](https://ollama.com) for fully local LLM inference
- ~500MB disk for the ONNX embedding model (downloaded once automatically)
- For voice mode: Chrome or Edge

---

## Installation

```bash
git clone https://github.com/DSWagner/psycho_portal
cd psycho_portal
python -m venv venv
# Windows: ./venv/Scripts/activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

## Quick Start

```bash
python main.py serve   # Web UI at http://localhost:8000
python main.py chat    # Rich terminal dashboard
```

---

## Personality System in Action

### Via chat (TARS-style commands):
```
"set humor to 90%"              → Humor: 75% → 90%
"be more direct"                → Directness: 88% → 100% (capped)
"dial down the sass"            → Sass: 38% → 18%
"set your directness to 100%"   → Directness: 100%
"be a bit less formal"          → Formality: 12% → 0% (capped)
```

### Via Web UI:
Click **⚙ Personality** in the header → drag sliders → Apply Changes

### Via REST API:
```bash
# Get current personality
GET /api/personality

# Update traits
PATCH /api/personality
{ "humor_level": 0.9, "sass_level": 0.6 }

# Set single trait
POST /api/personality/trait
{ "trait": "humor", "value": 0.9 }
```

---

## Proactive Features

### Reminders
Create via chat:
```
"remind me to submit the report tomorrow at 9am"
"set a reminder for the team meeting next Friday at 2pm"
"remind me in 30 minutes to take a break"
```

Or via API:
```bash
POST /api/reminders
{ "title": "Submit report", "due_timestamp": 1234567890, "priority": "high" }

GET /api/reminders          # List pending
PATCH /api/reminders/{id}/complete
PATCH /api/reminders/{id}/snooze?minutes=15
```

### Calendar
```bash
POST /api/calendar
{ "title": "Team standup", "start_timestamp": 1234567890, "location": "Zoom" }

GET /api/calendar           # Upcoming 7 days
GET /api/calendar/today     # Today's events
```

### Notifications
The web UI polls `GET /api/notifications` every 30 seconds.
The 🔔 bell shows unread count. Click to see all notifications.

### Check-ins
The agent checks in automatically:
- **Morning** (6–11am): "Good morning [name] — I see you've got a busy one ahead..."
- **Evening** (6–11pm): references what you worked on, asks how the day went
- **Long gap**: "Welcome back — it's been 3 days. Here's what was pending..."
- **Stress**: detects frustration signals from recent sessions, opens with care

---

## Voice Mode

1. Run `python main.py serve` → open `http://localhost:8000`
2. Click **🎤 Voice** → click the microphone
3. Speak naturally — transcript appears in real time
4. Agent responds and speaks back; mic reopens automatically

### TTS Options

| Provider | Quality | Cost | Config |
|----------|---------|------|--------|
| `browser` (default) | Good | **Free** | None |
| `openai` | High | ~$0.015/1k chars | `OPENAI_API_KEY` |
| `elevenlabs` | Highest | Paid | `ELEVENLABS_API_KEY` |
| `local` | Good | **Free** | `pyttsx3` or `kokoro-onnx` |

### STT Options

| Provider | Quality | Cost | Config |
|----------|---------|------|--------|
| `browser` (default) | Good | **Free** | None (Chrome/Edge required) |
| `whisper_local` | High | **Free** | `faster-whisper` installed |

---

## Full Local Mode (No API Keys)

PsychoPortal is designed to run 100% offline in production:

```env
# .env for fully-local setup
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2          # or mistral, qwen2.5, etc.

TTS_PROVIDER=local
LOCAL_TTS_BACKEND=pyttsx3       # zero download, system TTS
# or LOCAL_TTS_BACKEND=kokoro   # high quality, ~300MB download

STT_PROVIDER=whisper_local
WHISPER_MODEL=base              # ~145MB download
WHISPER_BACKEND=faster_whisper
```

Then:
```bash
ollama serve
ollama pull llama3.2
pip install pyttsx3 faster-whisper  # optional local models
python main.py serve
```

Embeddings (ChromaDB) already use ONNX/sentence-transformers — **fully local by default**.

---

## In-Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/stats` | Memory, graph, and session statistics |
| `/graph` | Inspect top knowledge graph nodes |
| `/tasks` | View pending tasks |
| `/reminders` | View pending reminders |
| `/health` | View logged health metrics |
| `/facts` | List stored facts with confidence scores |
| `/personality` | Show current personality calibration |
| `/ingest <path>` | Ingest a file or folder |
| `/reflect` | Run post-session reflection |
| `/mistakes` | Show recorded past mistakes |
| `/clear` | Clear the screen |
| `exit` / `quit` | Exit (triggers reflection automatically) |

---

## Web UI API Reference

Interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/personality` | Get current personality traits |
| `PATCH` | `/api/personality` | Update personality traits |
| `POST` | `/api/personality/trait` | Set a single trait |
| `GET` | `/api/notifications` | Get pending notifications |
| `POST` | `/api/notifications/{id}/read` | Mark notification as read |
| `GET` | `/api/reminders` | List pending reminders |
| `POST` | `/api/reminders` | Create a reminder |
| `PATCH` | `/api/reminders/{id}/complete` | Complete a reminder |
| `PATCH` | `/api/reminders/{id}/snooze` | Snooze a reminder |
| `GET` | `/api/calendar` | Get upcoming events |
| `GET` | `/api/calendar/today` | Get today's events |
| `POST` | `/api/calendar` | Create a calendar event |
| `DELETE` | `/api/calendar/{id}` | Delete a calendar event |
| `GET` | `/api/voice/config` | Active TTS/STT provider info |
| `POST` | `/api/voice/tts` | Text-to-speech (returns audio) |
| `POST` | `/api/voice/stt` | Speech-to-text (local Whisper) |
| `WS` | `/ws/chat` | Streaming WebSocket chat |
| ... | *All previous endpoints* | See /docs for full reference |

---

## Configuration Reference (`.env`)

```env
# ── LLM Provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER=anthropic           # "anthropic" or "ollama"
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# ── Local Model (Ollama) ──────────────────────────────────────────────────────
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2

# ── Personality (TARS-style, 0.0–1.0) ────────────────────────────────────────
# PERSONALITY_HUMOR=0.75
# PERSONALITY_DIRECTNESS=0.88
# PERSONALITY_WARMTH=0.72
# PERSONALITY_WIT=0.82
# PERSONALITY_SASS=0.38
# PERSONALITY_FORMALITY=0.12
# PERSONALITY_PROACTIVE=0.82
# PERSONALITY_EMPATHY=0.78

# ── Proactive System ──────────────────────────────────────────────────────────
# PROACTIVE_ENABLED=true
# CHECKIN_ENABLED=true
# GOOGLE_CALENDAR_CREDENTIALS=data/google_credentials.json

# ── TTS / STT ─────────────────────────────────────────────────────────────────
# TTS_PROVIDER=browser          # browser | openai | elevenlabs | local
# STT_PROVIDER=browser          # browser | whisper_local
# LOCAL_TTS_BACKEND=pyttsx3     # pyttsx3 | kokoro | coqui
# WHISPER_MODEL=base            # tiny | base | small | medium | large-v3

# ── Web Search ────────────────────────────────────────────────────────────────
# WEB_SEARCH_ENABLED=true
# BRAVE_API_KEY=                # optional

# ── Storage ───────────────────────────────────────────────────────────────────
# DATA_DIR=data
# DB_PATH=data/psycho.db
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
│   ├── personality/              ← TARS-style personality engine
│   │   ├── traits.py             ← AgentPersonality (9 adjustable traits)
│   │   ├── user_profile.py       ← Dynamic user personality model
│   │   └── adapter.py            ← Prompt section generator
│   ├── proactive/                ← Proactive agent systems
│   │   ├── reminders.py          ← Smart reminder manager + NL time parsing
│   │   ├── calendar_manager.py   ← Local calendar + Google Calendar sync
│   │   ├── checkin.py            ← Context-aware check-in logic
│   │   └── scheduler.py          ← Background async scheduler
│   ├── llm/                      ← LLM abstraction (Anthropic + Ollama + local)
│   │   ├── whisper_local.py      ← Local Whisper STT provider
│   │   └── local_tts.py          ← Local TTS (pyttsx3 / Kokoro / Coqui)
│   ├── memory/                   ← 4-tier memory (short, long, semantic, episodic)
│   ├── knowledge/                ← graph engine, extractor, evolver, reasoner
│   ├── learning/                 ← mistake tracker, signal detector, journal
│   ├── tools/                    ← pluggable agent tools (web_search.py)
│   ├── domains/                  ← coding, health, tasks, general + router
│   ├── storage/                  ← SQLite, ChromaDB, graph JSON store
│   ├── cli/                      ← Rich TUI, chat view, dashboard
│   └── api/                      ← FastAPI server, WebSocket, web UI
│       ├── routes/
│       │   ├── chat.py
│       │   ├── graph.py
│       │   ├── tasks.py
│       │   ├── health_metrics.py
│       │   ├── voice.py          ← TTS + local Whisper STT
│       │   └── personality.py    ← personality, notifications, reminders, calendar
│       └── static/
│           └── index.html        ← single-page web UI
│
└── data/                         ← all personal data (gitignored)
    ├── psycho.db                 ← SQLite: interactions, facts, tasks, reminders, calendar
    ├── personality.json          ← saved personality trait levels
    ├── graph/                    ← knowledge graph
    ├── vectors/                  ← ChromaDB embeddings
    ├── journals/                 ← session journals
    └── logs/
```

---

## Phase Roadmap

| Phase | Status | Feature |
|-------|--------|---------|
| 1 | ✅ Done | Foundation: agent core, 4-tier memory, Rich CLI |
| 2 | ✅ Done | Semantic memory (ChromaDB + ONNX embeddings) |
| 3 | ✅ Done | Knowledge graph + file ingestion |
| 4 | ✅ Done | Self-evolution: reflection, mistake tracker, insights |
| 5 | ✅ Done | Domain intelligence: coding execution, health, tasks |
| 6 | ✅ Done | FastAPI server + streaming WebSocket + web UI |
| 7 | ✅ Done | Web UI v2: session history, file upload, image vision |
| 8 | ✅ Done | Voice mode: STT + TTS + animated blob UI |
| 9 | ✅ Done | Graph explorer, web search injection, inline image chat |
| **10** | ✅ **Done** | **TARS/Jarvis personality engine + user personality learning** |
| **11** | ✅ **Done** | **Proactive system: reminders, calendar, check-ins, notifications** |
| **12** | ✅ **Done** | **Full local model stack: Whisper STT + local TTS + Ollama** |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| LLM (API) | Claude Haiku/Sonnet/Opus |
| LLM (local) | Ollama (any GGUF model) |
| Vision | Claude Vision API |
| Personality | Custom TARS-style trait system |
| Knowledge graph | NetworkX + confidence scoring |
| Vector store | ChromaDB (local, no Docker) |
| Embeddings | all-MiniLM-L6-v2 (ONNX, CPU-ready) |
| Database | SQLite + aiosqlite |
| CLI | Rich + Click + prompt_toolkit |
| Web API | FastAPI + uvicorn |
| Web UI | Vanilla JS + D3.js + marked.js |
| Voice STT | Browser Web Speech API / Local Whisper |
| Voice TTS | Browser / OpenAI / ElevenLabs / Local (pyttsx3/Kokoro) |
| Scheduler | asyncio-based background task |
| Calendar | Local SQLite / optional Google Calendar API |
| Config | pydantic-settings |

---

## License

MIT — do whatever you want with it.

---

*Built with Claude Sonnet 4.6*
