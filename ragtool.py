"""Constitution PDF RAG using a dedicated Gemini embedding model.

Key fixes over the original:
    * `GEMINI_EMBEDDING_MODEL` is separate from the chat model.
    * `_ensure_indexed` writes a `sha256` marker so re-indexing runs when the
      PDF changes, and does not rely on Chroma's private `_collection` counter
      as the sole signal.
    * Import from `langchain_chroma` when available; fall back to the
      deprecated `langchain_community.vectorstores.Chroma` so existing
      environments do not break.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pymupdf
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

log = logging.getLogger(__name__)

try:  # Prefer the new dedicated package.
    from langchain_chroma import Chroma  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent import.
    from langchain_community.vectorstores import Chroma  # type: ignore


# Constants preserved as module attributes so tests can monkeypatch them.
GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_EMBEDDING_MODEL = config.GEMINI_EMBEDDING_MODEL
GEMINI_API_KEY = config.GEMINI_API_KEY
HTTP_TIMEOUT_SECONDS = config.HTTP_TIMEOUT_SECONDS
DEFAULT_PDF_PATH = config.DEFAULT_PDF_PATH
PDF_PATH = config.CONSTITUTION_PDF_PATH
PERSIST_DIRECTORY = Path(__file__).resolve().parent / "constitution_chroma_db"
COLLECTION_NAME = "indian_constitution"
DEFAULT_QUERY = "The Constitution of India"
DEFAULT_TOP_K = 4


def _configured_path() -> Path:
    """Preserved for tests; returns the effective configured PDF path."""
    return config.env_path("CONSTITUTION_PDF_PATH", config.DEFAULT_PDF_PATH)


def _load_pdf_pages() -> list[Document]:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Constitution PDF not found at {PDF_PATH}. "
            "Set CONSTITUTION_PDF_PATH to an existing PDF file."
        )
    pdf = pymupdf.open(PDF_PATH)
    pages: list[Document] = []
    try:
        for page_number, page in enumerate(pdf):
            text = page.get_text("text")
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(PDF_PATH), "page": page_number},
                    )
                )
    finally:
        pdf.close()
    return pages


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embedding_model(),
        persist_directory=str(PERSIST_DIRECTORY),
    )


def _pdf_hash() -> str:
    if not PDF_PATH.is_file():
        return ""
    hasher = hashlib.sha256()
    with PDF_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _marker_path() -> Path:
    return PERSIST_DIRECTORY / ".indexed_sha256"


def _needs_index(vectorstore: Chroma) -> bool:
    """Return True when the store is empty or the PDF hash changed."""
    try:
        count = vectorstore._collection.count()  # noqa: SLF001 - Chroma has no public helper
    except Exception:  # noqa: BLE001
        count = 0
    if count == 0:
        return True
    marker = _marker_path()
    if not marker.is_file():
        return True
    return marker.read_text(encoding="utf-8").strip() != _pdf_hash()


def _is_transient_api_error(exc: BaseException) -> bool:
    """True for Gemini rate-limit / server-side errors that warrant a retry."""
    text = str(exc)
    return bool(
        re.search(r"\b(?:429|50[0-9])\b", text)
        or "RESOURCE_EXHAUSTED" in text
        or "UNAVAILABLE" in text
        or "rate limit" in text.lower()
        or "quota" in text.lower()
    )


def _suggested_retry_seconds(exc: BaseException) -> Optional[float]:
    """Honour the server's suggested wait delay when the API includes one."""
    text = str(exc)
    match = re.search(r"retry(?:Delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


EMBED_BATCH_SIZE = 16
EMBED_BATCH_QUIET_SECONDS = 0.8
EMBED_MAX_RETRIES = 5


def _add_documents_with_retries(vectorstore: Chroma, documents: list[Document]) -> None:
    """Add one batch of documents, retrying on transient 429/5xx embed errors."""
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            vectorstore.add_documents(documents)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == EMBED_MAX_RETRIES or not _is_transient_api_error(exc):
                raise
            wait = _suggested_retry_seconds(exc) or min(60.0, 10.0 * attempt)
            log.warning(
                "Embedding attempt %d/%d failed (%s). Waiting %.1fs.",
                attempt, EMBED_MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)


def _ensure_indexed(vectorstore: Chroma) -> None:
    if not _needs_index(vectorstore):
        return

    pages = _load_pdf_pages()
    chunks = _split_documents(pages)
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        _add_documents_with_retries(vectorstore, chunks[start : start + EMBED_BATCH_SIZE])
        time.sleep(EMBED_BATCH_QUIET_SECONDS)

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(_pdf_hash(), encoding="utf-8")


def retrieve_constitution_chunks(query: str = DEFAULT_QUERY) -> str:
    """Retrieve the chunks most relevant to the caller's query."""
    try:
        vectorstore = _get_vectorstore()
        _ensure_indexed(vectorstore)
        results = vectorstore.similarity_search(query, k=DEFAULT_TOP_K)
    except Exception as exc:  # noqa: BLE001
        log.warning("RAG retrieval failed: %s", exc)
        return (
            "RAG retrieval is temporarily unavailable "
            f"(external API error: {exc})."
        )
    parts: list[str] = []
    for index, document in enumerate(results, start=1):
        page_num = document.metadata.get("page", "Unknown")
        if isinstance(page_num, int):
            page_num = page_num + 1
        parts.append(
            f"--- Result {index} ---\nPage: {page_num}\n{document.page_content.strip()}"
        )
    return "\n\n".join(parts)
