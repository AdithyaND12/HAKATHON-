# HAKATHON-

An interactive LangGraph chatbot powered by an OpenAI-compatible LM Studio server.
It combines web search, calculations, time lookup, stock prices, and retrieval from
the Constitution of India PDF. It can also repeat searches on a robust background
schedule while the CLI remains available for new requests.

## What's new (scheduler v2)

The scheduling engine was rewritten to fix the pain points of the original:

- **Cancel / list / pause / resume** — every job has a stable ID (`/jobs`, `/cancel <id>`).
- **Interleaved output fixed** — a global console lock serializes all output; per-job
  run history is written to `.hakathon/history/<id>/run-NNN.json`.
- **Schedules survive restart** — the job registry is snapshotted to
  `.hakathon/jobs.json` and unfinished schedules resume automatically next time you
  launch the CLI.
- **Retries with backoff** — a transient DuckDuckGo rate limit or connection blip no
  longer aborts the schedule; each run is retried with exponential backoff.
- **Absolute times** — natural-language prompts like *"search python news at 3pm"*,
  *"tomorrow 9am"*, or *"in 5 minutes"* are parsed and honored.
- **Improved fallback planner** — recognises *hourly*, *daily*, *every day at X*,
  *twice*, *thrice*, *in N minutes*.
- **Separate embedding model** — `LM_STUDIO_EMBEDDING_MODEL` fixes the #1 cause of
  Constitution RAG failing on first run (the chat model rarely supports embeddings).

## Features

- Local chat and tool-calling through LM Studio.
- Web search via DuckDuckGo (wrapped with retry + backoff).
- Calculator and current-time tools.
- Alpha Vantage stock-price lookup with structured error taxonomy.
- Constitution PDF retrieval using PyMuPDF, LangChain text splitting, an OpenAI-
  compatible embedding model, and Chroma.
- Natural-language scheduling: *"check tesla news every 10 minutes for 5 times"*,
  *"search AI news at 3pm"*, *"monitor python news hourly"*.
- Background scheduled jobs with cancellation, pause/resume, persistence, and
  no unnecessary wait after the final run.

## Requirements

- Python 3.10 or newer.
- [LM Studio](https://lmstudio.ai/) running an OpenAI-compatible local server.
- A chat model **and** a separate embedding model available in LM Studio.
- An Alpha Vantage API key if stock-price lookups are required.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` includes everything, including the new
`streamlit-autorefresh` dependency the UI uses for background polling.

If you'd rather install manually:

```bash
pip install \
  arrow chromadb ddgs langchain-community langchain-chroma langchain-core \
  langchain-openai langchain-text-splitters langgraph pydantic pymupdf \
  python-dotenv requests pytest streamlit streamlit-autorefresh
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `LM_STUDIO_MODEL` | Yes | `qwen2.5-coder-7b-instruct` | Chat model served by LM Studio. |
| `LM_STUDIO_EMBEDDING_MODEL` | For RAG | `nomic-embed-text-v1.5` | Embedding model (must be a real embedding model, not a chat model). |
| `LM_STUDIO_BASE_URL` | Yes | `http://localhost:1234/v1` | OpenAI-compatible LM Studio endpoint. |
| `LM_STUDIO_API_KEY` | No | `lm-studio` | Key accepted by the local server. |
| `ALPHAVANTAGE_API_KEY` | For stocks | — | Alpha Vantage API key. |
| `CONSTITUTION_PDF_PATH` | No | `pdfs/c9fe9c9b6840524844316f74bb1c556c.pdf` | PDF path (relative or absolute). |
| `HTTP_TIMEOUT_SECONDS` | No | `10` | Timeout for stock API requests. |
| `WAIT_MAX_SECONDS` | No | `3600` | Upper bound for a single `wait` / interval. |
| `MAX_AUTO_RUNS` | No | `20` | Cap on the number of runs the planner can schedule. |
| `HAKATHON_DATA_DIR` | No | `.hakathon` | Where job registry + run history live. |
| `DUCKDUCKGO_REGION` | No | `us-en` | DuckDuckGo region parameter. |
| `SEARCH_MAX_RETRIES` | No | `3` | Retries for a single web_search invocation. |
| `SEARCH_RETRY_BACKOFF_SECONDS` | No | `2.0` | Base backoff (doubles each retry). |

The local `.env` file is ignored by Git. Never commit real API keys.

## Running the chatbot (Streamlit UI)

The primary interface is now a Streamlit web app with a dark, terminal-inspired
aesthetic. Start LM Studio's local server with the configured models loaded, then:

```bash
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The interface is
intentionally minimal: one chat window. Scheduling still works — just describe
it in natural language.

### Example prompts

```
What are the latest technology headlines?
Calculate 27 times 14.
What does the Constitution say about freedom of speech?
What is the current price of AAPL?
Check the latest Python news every 1 minute for 3 times.
Monitor tesla stock hourly for 5 times.
Search AI news tomorrow at 9am.
Search python releases at 3pm.
```

Started a schedule? Each completed run appears back in the chat automatically
as an amber-highlighted message. Use the collapsed sidebar (top-left `»`) to
view active schedules, clear the chat, or cancel every running job.

### Legacy CLI (optional)

If you prefer the terminal, the old CLI still works:

```bash
python app.py
```

Slash commands: `/help /jobs /cancel <id> /pause <id> /resume <id> /logs <id>
/clear`.

## Persistence

- `.hakathon/jobs.json` — atomic snapshot of every registered job. Unfinished
  jobs are automatically resumed on the next `python app.py`.
- `.hakathon/history/<job-id>/run-NNN.json` — the assistant output for each run,
  timestamped.

## Constitution retrieval

The first Constitution query indexes the configured PDF into the local
`constitution_chroma_db/` directory. A `sha256` marker is written so re-indexing
runs automatically whenever the PDF changes. `LM_STUDIO_EMBEDDING_MODEL` must be a
real embedding model (e.g., `nomic-embed-text-v1.5`, `bge-small-en-v1.5`).

## Tests

```bash
pytest -q
```

Tests mock every network + LLM call, so no LM Studio or Alpha Vantage key is needed.
The suite covers: config parsing, stock error taxonomy, RAG configuration, planner
regex fallback (including absolute times), and the full scheduler (retries,
persistence, resume, pause, cancel, no-wait-after-final-run).

## Project structure

```
streamlit_app.py      Primary UI — dark terminal-inspired chat (Streamlit)
app.py                Legacy CLI + LangGraph wiring + tools (imported by streamlit_app)
config.py             Centralised env parsing
planner.py            Search + schedule planner (LLM + regex fallback + absolute time)
scheduler.py          Job registry, persistence, retries, pause/resume
tools_search.py       DuckDuckGo tool with retry / backoff
ragtool.py            Constitution PDF indexing + retrieval
.streamlit/config.toml Dark theme config
tests/                pytest suite
pdfs/                  Default Constitution PDF
.hakathon/            Runtime data (jobs.json, per-job history) — git-ignored
constitution_chroma_db/  Vector store — git-ignored
```

## Troubleshooting

- **LM Studio connection error**: confirm the server is running and
  `LM_STUDIO_BASE_URL` matches its endpoint.
- **Constitution RAG errors like "not a valid embedding model"**: set
  `LM_STUDIO_EMBEDDING_MODEL` to an actual embedding model loaded in LM Studio.
- **Stock lookup unavailable**: set `ALPHAVANTAGE_API_KEY` in `.env`; the rest of
  the chatbot remains usable without it.
- **A schedule seems stuck**: run `/jobs` to see its state (`pending`, `running`,
  `paused`, `completed`, `failed`), then `/cancel <id>` if needed. Check
  `.hakathon/history/<id>/` for per-run output.

## License

MIT
