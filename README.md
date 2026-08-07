# HAKATHON-

An interactive LangGraph chatbot powered by an OpenAI-compatible LM Studio server. It combines web search, calculations, time lookup, stock prices, and retrieval from the Constitution of India PDF. It can also repeat searches on a background schedule while the CLI remains available for new requests.

## Features

- Local chat and tool-calling through LM Studio.
- Web search with DuckDuckGo when current information is needed.
- Calculator and current-time tools.
- Alpha Vantage stock-price lookup with configurable timeouts and clear API errors.
- Constitution PDF retrieval using PyMuPDF, LangChain text splitting, OpenAI-compatible embeddings, and Chroma.
- Natural-language scheduling such as “check this every 5 minutes for 3 times”.
- Background scheduled jobs with cancellation and no unnecessary wait after the final run.

## Requirements

- Python 3.10 or newer.
- [LM Studio](https://lmstudio.ai/) running an OpenAI-compatible local server.
- A model available in LM Studio. The configured model is used for chat and embeddings.
- An Alpha Vantage API key if stock-price lookups are required.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install \
  arrow \
  chromadb \
  duckduckgo-search \
  langchain-community \
  langchain-core \
  langchain-openai \
  langchain-text-splitters \
  langgraph \
  pydantic \
  pymupdf \
  python-dotenv \
  requests
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Configuration

Copy the example configuration and edit it as needed:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `LM_STUDIO_MODEL` | Yes | `qwen2.5-coder-7b-instruct` | Model served by LM Studio. |
| `LM_STUDIO_BASE_URL` | Yes | `http://localhost:1234/v1` | OpenAI-compatible LM Studio endpoint. |
| `LM_STUDIO_API_KEY` | No | `lm-studio` | Key accepted by the local server. |
| `ALPHAVANTAGE_API_KEY` | For stock prices | — | Alpha Vantage API key. |
| `CONSTITUTION_PDF_PATH` | No | `pdfs/c9fe9c9b6840524844316f74bb1c556c.pdf` | Constitution PDF path, relative to the project directory or an absolute path. |
| `HTTP_TIMEOUT_SECONDS` | No | `10` | Timeout for stock API requests. |

The local `.env` file is ignored by Git. Never commit real API keys.

## Running the chatbot

Start LM Studio’s local server with the configured model loaded, then run:

```bash
python app.py
```

Enter a normal question at the `You:` prompt. Examples:

```text
What are the latest technology headlines?
Calculate 27 times 14.
What does the Constitution say about freedom of speech?
What is the current price of AAPL?
Check the latest Python news every 1 minute for 3 times.
```

Type `exit`, `quit`, or `q` to leave. Scheduled searches start in a background worker, so the CLI can accept another request immediately. Active schedules are asked to stop when the CLI exits.

## Constitution retrieval

The first Constitution query indexes the configured PDF into the local `constitution_chroma_db/` directory. That generated database is ignored by Git. If the PDF is moved, set `CONSTITUTION_PDF_PATH` before starting the application; the program reports a clear error when the configured file does not exist.

## Tests

Run the test suite from the project directory:

```bash
pytest -q
```

The tests mock network and language-model calls, so they do not require a running LM Studio server or a live Alpha Vantage key.

## Project structure

```text
app.py                         CLI, LangGraph workflow, tools, and scheduler
ragtool.py                     Constitution PDF indexing and retrieval
pdfs/                          Default Constitution PDF
tests/                         Configuration, HTTP, PDF, and scheduler tests
.env.example                   Safe configuration template
```

## Troubleshooting

- **LM Studio connection error:** confirm the server is running and `LM_STUDIO_BASE_URL` matches its endpoint.
- **Embedding error:** ensure the configured LM Studio model supports embeddings and is loaded by the server.
- **Stock lookup unavailable:** set `ALPHAVANTAGE_API_KEY` in `.env`; the rest of the chatbot remains usable without it.
- **PDF not found:** verify `CONSTITUTION_PDF_PATH` and use a path relative to the project directory or an absolute path.
