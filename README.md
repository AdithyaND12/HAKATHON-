# HAKATHON-

An interactive LangGraph chatbot powered by Google Gemini.
It combines web search, calculations, time lookup, stock prices, and RAG retrieval
from PDFs the user uploads (each indexed into its own vector store). It can also
repeat searches on a robust background schedule while the CLI remains available
for new requests.

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
- **Separate embedding model** — `JINA_EMBEDDING_MODEL` keeps RAG aligned with
  Jina's free embedding API (`jina-embeddings-v4`, 1M free tokens/day); the
  chat LLM remains Gemini.
- **Embedding-swap safety** — index markers record the embedding model + PDF
  hash, so switching embedding models automatically wipes and rebuilds any
  incompatible (old-dimension) Chroma collections.

## Features

- Local chat and tool-calling through Gemini.
- Web search via DuckDuckGo (wrapped with retry + backoff).
- Calculator and current-time tools.
- Alpha Vantage stock-price lookup with structured error taxonomy.
- Uploaded-PDF retrieval using PyMuPDF, LangChain text splitting, a Jina
  embedding model, and Chroma — no built-in document; the user uploads the PDF.
- Natural-language scheduling: *"check tesla news every 10 minutes for 5 times"*,
  *"search AI news at 3pm"*, *"monitor python news hourly"*.
- Background scheduled jobs with cancellation, pause/resume, persistence, and
  no unnecessary wait after the final run.

## Requirements

- Python 3.10 or newer.
- A Gemini API key (chat model).
- A Jina API key for the free embedding API (https://jina.ai/embeddings/).
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
  langchain-google-genai langchain-text-splitters langgraph pydantic pymupdf \
  python-dotenv requests pytest streamlit streamlit-autorefresh
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key. |
| `GEMINI_MODEL` | Yes | `gemini-3.5-flash-lite` | Chat model. |
| `JINA_API_KEY` | For RAG | — | Jina Embeddings API key (free tier: https://jina.ai/embeddings/). |
| `JINA_EMBEDDING_MODEL` | No | `jina-embeddings-v4` | Free embedding model used for all RAG indexing + search. |
| `JINA_EMBEDDING_BATCH_SIZE` | No | `100` | Chunks per embedding request (one POST per batch). |
| `ALPHAVANTAGE_API_KEY` | For stocks | — | Alpha Vantage API key. |
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
aesthetic. Set your Gemini env vars, then:

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
What does the Constitution say about freedom of speech? (with that PDF uploaded)
What is the current price of AAPL?
Check the latest Python news every 1 minute for 3 times.
Monitor tesla stock hourly for 5 times.
Search AI news tomorrow at 9am.
Search python releases at 3pm.
```

Started a schedule? Each completed run appears back in the chat automatically
as an amber-highlighted message. Use the collapsed sidebar (top-left `»`) to
view active schedules, clear the chat, or cancel every running job.

## Scheduling function

Schedules are created programmatically through `Scheduler.start()` (or the
backward-compatible `app.run_scheduled_search()` helper) — identified by
`SearchPlan` from the planner:

```python
from app import run_scheduled_search

job = run_scheduled_search(
    prompt="check python news every 10 minutes for 5 times",
    interval_minutes=10.0,     # Optional[float] — gap between runs (None = no repeat)
    run_count=5,               # Optional[int] — how many runs total (None = unlimited)
    search_query="python news",  # Optional[str] — fallback query for the search tool
    absolute_start_iso="2026-08-14T09:00:00",  # Optional[str] — first run at a fixed time
    task_type="search",        # "search" | "rag" | "reminder" | "calculation" | "chat"
    reminder_text=None,        # required when task_type="reminder"
)
print(job.id)  # stable job id for /jobs, /cancel, /pause, /resume
```

The equivalent `Scheduler.start(**kwargs)` (scheduler.py) accepts the same
arguments plus an optional `job_id`. Examples of what the planner accepts as
natural language:

- `"check tesla news every 10 minutes for 5 times"` → interval 10m, 5 runs
- `"search AI news at 3pm"` → a single run at 15:00 local time
- `"monitor python news hourly"` → interval 60m, run forever (no run_count)
- `"remind me to drink water in 5 minutes"` → one reminder in 5 minutes

Every job runs in its own background worker thread, persists to
`.hakathon/jobs.json`, and can be controlled from the CLI with `/jobs`,
`/cancel <id>`, `/pause <id>`, `/resume <id>`, and `/logs <id>`.

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

## RAG document retrieval

There is no built-in document: the user uploads PDFs in the Streamlit sidebar,
and each is indexed into its own hash-derived collection under
`chroma_stores/<embedding-model>/` (e.g. `chroma_stores/jina-embeddings-v4/`).
The LLM's `get_rag_chunks` tool queries whichever document is selected as active.
A marker (embedding model + `sha256`) is written so re-indexing runs automatically
whenever the PDF changes **or the embedding model changes** — old vectors from a
different model are never reused, they are wiped and rebuilt. `JINA_API_KEY` must
be set; the embedding model defaults to `jina-embeddings-v4`.

## Tests

```bash
pytest -q
```

Tests mock every network + LLM call, so no Gemini or Alpha Vantage key is needed.
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
ragtool.py            User-uploaded PDF indexing + retrieval
.streamlit/config.toml Dark theme config
tests/                pytest suite
.hakathon/            Runtime data (jobs.json, per-job history) — git-ignored
chroma_stores/       Vector stores (one dir per embedding model) — git-ignored
```

## Troubleshooting

- **Gemini authentication error**: confirm `GEMINI_API_KEY` is set correctly in
  your `.env`.
- **Embedding API error (RAG)**: confirm `JINA_API_KEY` is set correctly in your
  `.env` (get a free key at https://jina.ai/embeddings/).
- **Free-tier embedding limits**: the Jina free tier covers 1M tokens/day at
  100 requests/minute, and `jina-embeddings-v4` is additionally throttled by
  design. If a batch fails on rate limits the retry/backoff handles it; if your
  day's tokens run out, indexing resumes automatically after the reset.
- **Embedding model changed?** Nothing to do — stores are namespaced per
  embedding model (`chroma_stores/<model>/`) and old-dimension vectors are
  detected as incompatible and rebuilt automatically on next use.
- **"Chroma collection not initialized"** — this happens when a Chroma store is
  left in a broken state (e.g. an interrupted embedding-model switch). The app
  now detects it, wipes the affected store, and rebuilds automatically. To
  reset manually: stop the app and delete `chroma_stores/`.
- **Stock lookup unavailable**: set `ALPHAVANTAGE_API_KEY` in `.env`; the rest of
  the chatbot remains usable without it.
- **A schedule seems stuck**: run `/jobs` to see its state (`pending`, `running`,
  `paused`, `completed`, `failed`), then `/cancel <id>` if needed. Check
  `.hakathon/history/<id>/` for per-run output.

## License

MIT
