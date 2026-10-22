"""PDF RAG using Jina's free embedding API.

There is no built-in document: the user uploads PDFs (via the web UI), and
each is indexed into its own hash-derived Chroma collection so the LLM's
`get_rag_chunks` tool can query whichever document the user has selected as
active.

Key design points:
    * Embeddings come from Jina (`JinaEmbeddings`, free tier); the chat LLM
      is untouched and stays a Gemini model.
    * Index markers record BOTH the embedding model and the PDF sha256, so
      switching embedding models automatically invalidates the old vectors
      (vectors from a different model have a different dimension and are
      incompatible with a Chroma collection built by Jina) and triggers a safe
      rebuild.
    * Import from `langchain_chroma` when available; fall back to the
      deprecated `langchain_community.vectorstores.Chroma` so existing
      environments do not break.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pymupdf
import requests
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    GoogleGenerativeAIEmbeddings = None  # type: ignore[assignment,misc]

import config

log = logging.getLogger(__name__)

try:  # Prefer the new dedicated package.
    from langchain_chroma import Chroma  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent import.
    from langchain_community.vectorstores import Chroma  # type: ignore


# Constants preserved as module attributes so tests can monkeypatch them.
GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_API_KEY = config.GEMINI_API_KEY
JINA_API_KEY = config.JINA_API_KEY
JINA_RERANK_MODEL = config.JINA_RERANK_MODEL
RERANK_ENABLED = config.RERANK_ENABLED
RERANK_TIMEOUT_SECONDS = config.RERANK_TIMEOUT_SECONDS
JINA_EMBEDDING_MODEL = config.JINA_EMBEDDING_MODEL
EMBEDDING_PROVIDER = config.EMBEDDING_PROVIDER
GOOGLE_EMBEDDING_MODEL = config.GOOGLE_EMBEDDING_MODEL
HTTP_TIMEOUT_SECONDS = config.HTTP_TIMEOUT_SECONDS
JINA_EMBED_BATCH_SIZE = config.JINA_EMBEDDING_BATCH_SIZE
GOOGLE_EMBED_BATCH_SIZE = config.GOOGLE_EMBEDDING_BATCH_SIZE

# Active provider can be changed at runtime via set_embedding_provider().
_active_provider = EMBEDDING_PROVIDER


def get_embedding_provider() -> str:
    """Return the currently active embedding provider name."""
    return _active_provider


def get_embedding_model_name() -> str:
    """Return the model name for the currently active embedding provider."""
    if _active_provider == "google":
        return GOOGLE_EMBEDDING_MODEL
    return JINA_EMBEDDING_MODEL


def set_embedding_provider(provider: str) -> None:
    """Switch the embedding provider at runtime ('jina' or 'google').

    Resets the shared Chroma client so the next operation uses a store
    namespaced to the new provider+model.
    """
    global _active_provider, _CLIENT, PERSIST_DIRECTORY, _ACTIVE_COLLECTION
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(
            "Embedding provider must be 'jina' or 'google'."
        )
    provider = provider.strip().lower()
    if provider not in ("jina", "google"):
        raise ValueError(f"Unknown embedding provider: {provider!r} (expected 'jina' or 'google')")
    if provider == _active_provider:
        return
    _active_provider = provider
    _BM25_CACHE.clear()
    # Active IDs are namespaced to the old store; do not leak them into the
    # new provider, where they would silently resolve to an empty collection.
    _ACTIVE_COLLECTION = None
    global _ACTIVE_RAW
    _ACTIVE_RAW = None
    # Keep the module-level directory in sync: it is the single source of
    # truth used by every marker/meta path and can be overridden by callers.
    PERSIST_DIRECTORY = STORE_ROOT / _active_model_slug()
    # Reset client so it is re-created inside the new provider's store.
    _CLIENT = None


def _model_slug(model_name: str) -> str:
    """Filesystem-safe name for a model, e.g. ``jina-embeddings-v4``."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_name.lower()).strip("-._")
    return slug or "default"


# One persist directory PER embedding model: vectors from a different model
# (different dimension) are incompatible, and a stale Chroma store makes the
# new model's collections query-time-unusable. Namespacing means switching
# embedding models can never touch another model's store.
STORE_ROOT = Path(__file__).resolve().parent / "chroma_stores"
# One persist directory PER embedding model: vectors from a different model
# (different dimension) are incompatible. Namespacing by provider+model means
# switching embedding models can never touch another model's store.
def _active_model_slug() -> str:
    if _active_provider == "google":
        return _model_slug(GOOGLE_EMBEDDING_MODEL)
    return _model_slug(JINA_EMBEDDING_MODEL)

PERSIST_DIRECTORY = STORE_ROOT / _active_model_slug()


def _persist_dir() -> Path:
    """Current persist directory, overridable for tests and deployments."""
    return PERSIST_DIRECTORY
DEFAULT_QUERY = "the user's document"
DEFAULT_TOP_K = 4


def _embed_batch_size() -> int:
    """Chunks per embedding batch, provider-aware."""
    if _active_provider == "google":
        return GOOGLE_EMBED_BATCH_SIZE
    return JINA_EMBED_BATCH_SIZE


def _ocr_page(page) -> str:
    """OCR a scanned (image-only) page via tesseract through PyMuPDF.

    Returns the recognized text, or "" when OCR is unavailable or fails so
    the page is simply skipped instead of aborting the whole index.
    """
    try:
        return page.get_textpage_ocr(full=True, language="eng", dpi=200).extractText() or ""
    except Exception:  # noqa: BLE001 - missing tesseract binary etc.
        log.warning("OCR unavailable for a page of %s; skipping it.", getattr(page, "number", "?"))
        return ""


# ---- Heading-aware extraction -------------------------------------------------
# PDFs usually have no machine-readable structure, but font sizes are a strong
# heuristic: lines noticeably larger than the page's body text (or bold at body
# size) are headings. Recording them lets the chunker keep sections intact and
# attach a heading path to every chunk, which dramatically improves retrieval
# for questions about named sections and gives the LLM a precise citation.

_HEADING_SIZE_MULTIPLIER = 1.2  # size >= body * this -> a heading line
_BOLD_FLAG = 2**4  # PyMuPDF span "flags" bit 4 means bold
# Transient metadata key carrying heading candidates between page extraction
# and chunking; never persisted to the vector store.
_HEADINGS_META_KEY = "_headings"


def _page_heading_candidates(page) -> list[tuple[int, str]]:
    """Detect heading lines on a text-layer page as (level, text) pairs.

    Body font size is the mode of span sizes weighted by text length. A line
    is a heading when its size is >= 1.2x the body size, or when it is bold at
    at least body size. Levels are relative ranks within the page: the largest
    heading is level 1, smaller ones deeper. Returns [] for pages without a
    text layer (scanned pages come back from OCR with no font information).
    """
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001 - any extraction failure means no headings
        return []

    spans = [span for block in data.get("blocks", []) for line in block.get("lines", []) for span in line.get("spans", [])]
    spans = [s for s in spans if s.get("text", "").strip()]
    if not spans:
        return []

    weighted: dict[float, int] = {}
    for span in spans:
        weighted[span["size"]] = weighted.get(span["size"], 0) + len(span["text"])
    body_size = max(weighted, key=weighted.get)

    candidates: list[tuple[float, str]] = []
    for line in data.get("blocks", []):
        for l in line.get("lines", []):
            spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            size = max(s["size"] for s in spans)
            bold = any(s["flags"] & _BOLD_FLAG for s in spans)
            if size >= body_size * _HEADING_SIZE_MULTIPLIER or (bold and size >= body_size):
                text = " ".join(s["text"].strip() for s in spans).strip()
                if text:
                    candidates.append((size, text))

    levels: list[tuple[int, str]] = []
    for rank, (_, text) in enumerate(sorted(candidates, key=lambda c: -c[0])):
        levels.append((rank + 1, text))
    # Restore document order (positions within the page).
    positions = {text: idx for idx, (_, text) in enumerate(candidates)}
    levels.sort(key=lambda pair: positions[pair[1]])
    return levels


def _load_pdf_pages_from(pdf_path: Path) -> list[Document]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found at {pdf_path}.")
    pdf = pymupdf.open(pdf_path)
    pages: list[Document] = []
    ocr_used = False
    try:
        for page_number, page in enumerate(pdf):
            text = page.get_text("text")
            headings = _page_heading_candidates(page) if text.strip() else []
            if not text.strip():
                # Scanned PDF: no text layer, so recognize the page image.
                text = _ocr_page(page)
                ocr_used = ocr_used or bool(text.strip())
            if text.strip():
                metadata: dict = {"source": str(pdf_path), "page": page_number}
                if headings:
                    metadata[_HEADINGS_META_KEY] = headings
                pages.append(Document(page_content=text, metadata=metadata))
    finally:
        pdf.close()
    if ocr_used:
        log.info("OCR was used to extract text from scanned pages of %s.", pdf_path)
    return pages


def _split_documents(documents: list[Document]) -> list[Document]:
    """Split page documents into chunks, preserving heading context.

    Heading candidates detected during page extraction are threaded through
    the document in page order (standard outline algorithm: a heading of
    level N closes every deeper heading). Each chunk records both the leaf
    heading and the full heading path, so embedding/retrieval can place the
    chunk in its section instead of a 500-char island.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks: list[Document] = []
    chain: list[tuple[int, str]] = []
    for document in documents:
        headings = document.metadata.get(_HEADINGS_META_KEY, [])
        for level, text in headings:
            chain = [(l, h) for (l, h) in chain if l < level]
            chain.append((level, text))
        chunk_metadata = {
            key: value for key, value in document.metadata.items() if key != _HEADINGS_META_KEY
        }
        for piece in splitter.split_text(document.page_content):
            metadata = dict(chunk_metadata)
            if chain:
                metadata["heading"] = chain[-1][1]
                metadata["heading_path"] = " > ".join(text for _, text in chain)
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def _get_embedding_model() -> Embeddings:
    """Return the configured embedding model (Jina or Google)."""
    if _active_provider == "google":
        if GoogleGenerativeAIEmbeddings is None:
            raise ImportError(
                "langchain-google-genai is required for Google embeddings. "
                "Install it with: pip install langchain-google-genai"
            )
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
    # Default: Jina
    return JinaEmbeddings(
        model_name=JINA_EMBEDDING_MODEL,
        jina_api_key=JINA_API_KEY,
    )


_CLIENT: Optional[PersistentClient] = None
_CLIENT_PROVIDER: Optional[str] = None  # tracks which provider the client was created for


def _get_client() -> PersistentClient:
    """One shared persistent client per process.

    Chroma's writes go through an async write queue watched by the client.
    Creating a NEW client per call means a querying client may never see the
    other client's still-pending writes, causing spurious "collection not
    initialized" failures. A single shared client keeps index + query coherent.

    The client is reset when the embedding provider changes so the store
    directory is re-evaluated for the new provider+model namespace.
    """
    global _CLIENT, _CLIENT_PROVIDER
    current_dir = _persist_dir()
    if _CLIENT is None or _CLIENT_PROVIDER != _active_provider:
        current_dir.mkdir(parents=True, exist_ok=True)
        # allow_reset lets the corruption self-heal wipe the store in-process.
        _CLIENT = PersistentClient(
            path=str(current_dir), settings=Settings(allow_reset=True)
        )
        _CLIENT_PROVIDER = _active_provider
    return _CLIENT


def _get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embedding_model(),
        client=_get_client(),
    )


def _pdf_hash(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        return ""
    hasher = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_transient_api_error(exc: BaseException) -> bool:
    """True for provider rate-limit / server-side errors that warrant a retry."""
    text = str(exc)
    return bool(
        re.search(r"\b(?:429|50[0-9])\b", text)
        or "RESOURCE_EXHAUSTED" in text
        or "UNAVAILABLE" in text
        or "rate limit" in text.lower()
        or "quota" in text.lower()
    )


def _is_daily_quota_exhausted(exc: BaseException) -> bool:
    """True when the provider says the DAILY request quota is used up.

    These errors only clear after the daily reset, so retrying immediately is
    wasted time - callers fail fast instead of sleeping for minutes.
    """
    text = str(exc)
    return bool(
        re.search(r"free_tier|RequestsPerDay|quota(?:.*)per\s*day", text, re.IGNORECASE)
        and re.search(r"quota|RESOURCE_EXHAUSTED", text, re.IGNORECASE)
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


EMBED_BATCH_QUIET_SECONDS = 0.8
EMBED_MAX_RETRIES = 5


def _is_persist_corruption_error(exc: BaseException) -> bool:
    """True for chromadb errors that signal an unusable on-disk store.

    These can survive collection-level repair (e.g. stale dimension metadata
    or a half-written index after an embedding-model switch), so the caller
    escalates to a full store wipe + rebuild.
    """
    text = str(exc)
    return bool(
        re.search(r"not initialized|reset_collection", text, re.IGNORECASE)
        or "InvalidDimensionException" in text
        or "dimension" in text.lower() and "mismatch" in text.lower()
        or re.search(r"hnsw", text, re.IGNORECASE) is not None
    )


def _rebuild_store() -> None:
    """Reset the (model-specific) Chroma store so it starts clean.

    Uses `client.reset()` (never a directory delete) because the client's
    sqlite handle is open — deleting the folder underneath a live client
    produces a readonly-database error instead of a reset.
    """
    persist_dir = _persist_dir()
    log.warning("Resetting Chroma store %s and rebuilding from scratch.", persist_dir)
    _get_client().reset()
    _BM25_CACHE.clear()


def _add_documents_with_retries(vectorstore: Chroma, documents: list[Document]) -> None:
    """Add one batch of documents, retrying on transient 429/5xx embed errors."""
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            vectorstore.add_documents(documents)
            return
        except Exception as exc:  # noqa: BLE001
            if _is_daily_quota_exhausted(exc):
                log.error(
                    "Embedding stopped: the provider's daily request quota for "
                    "%s is exhausted (resets daily). Retrying cannot help: %s",
                    JINA_EMBEDDING_MODEL,
                    exc,
                )
                raise
            if attempt == EMBED_MAX_RETRIES or not _is_transient_api_error(exc):
                raise
            wait = _suggested_retry_seconds(exc) or min(60.0, 10.0 * attempt)
            log.warning(
                "Embedding attempt %d/%d failed (%s). Waiting %.1fs.",
                attempt, EMBED_MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)


# Bump whenever chunking/contextualization changes the meaning of stored
# vectors, so stale stores are rebuilt instead of served silently.
INDEX_PIPELINE_VERSION = 2


def _expected_marker(pdf_sha256: str) -> str:
    """Marker content = embedding provider + model + pipeline version + pdf hash.

    Including the provider and model means switching to a different embedding
    model or provider invalidates every old collection automatically,
    since vectors from a different model/dimension are incompatible. The
    pipeline version additionally invalidates collections whose vectors were
    produced by an older chunking/contextualization logic.
    """
    model = GOOGLE_EMBEDDING_MODEL if _active_provider == "google" else JINA_EMBEDDING_MODEL
    return f"{_active_provider}:{model}\n{INDEX_PIPELINE_VERSION}\n{pdf_sha256}"


def _format_results(results: list[Document]) -> str:
    parts: list[str] = []
    for index, document in enumerate(results, start=1):
        page_num = document.metadata.get("page", "Unknown")
        if isinstance(page_num, int):
            page_num = page_num + 1
        # Stored vectors carry a contextual prefix; show the raw passage and
        # the section path instead so the LLM quotes clean text with citations.
        content = document.metadata.get("raw_text") or document.page_content.strip()
        header = f"--- Result {index} ---\nPage: {page_num}"
        heading_path = document.metadata.get("heading_path")
        if heading_path:
            header += f"\nSection: {heading_path}"
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


# Fetch this many candidates over `k` for MMR diversity, since the top-k
# cosine hits are often near-duplicate passages of one paragraph.
MMR_FETCH_MULTIPLIER = 5

# ---- Hybrid retrieval: BM25 + vector + cross-encoder rerank ------------------

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
BM25_SIDECAR_VERSION = 1
RERANK_MAX_CANDIDATES = 20
BM25_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that "
    "the their this to was were will with".split()
)
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BM25_CACHE: dict[str, tuple[int, list[dict], "_BM25Index"]] = {}


def _bm25_tokenize(text: str) -> list[str]:
    return [
        token
        for token in _BM25_TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in BM25_STOPWORDS
    ]


def _chunk_key(raw_text: str, page: object) -> str:
    return hashlib.sha1(f"{page}|{raw_text}".encode("utf-8")).hexdigest()


def _document_key(document: Document) -> str:
    return _chunk_key(
        document.metadata.get("raw_text") or document.page_content,
        document.metadata.get("page"),
    )


class _BM25Index:
    """Dependency-free BM25 ranking over chunk texts."""

    def __init__(self, entries: list[dict]):
        self._term_freqs = []
        lengths = []
        for entry in entries:
            tokens = _bm25_tokenize(entry.get("raw_text") or "")
            self._term_freqs.append(Counter(tokens))
            lengths.append(len(tokens))
        self._lengths = lengths
        count = len(entries)
        self._avg_length = (sum(lengths) / count) if count else 0.0
        document_freq: Counter[str] = Counter()
        for freqs in self._term_freqs:
            document_freq.update(freqs.keys())
        self._idf = {
            term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def search(self, query: str, limit: int) -> list[int]:
        terms = _bm25_tokenize(query)
        if not terms or not self._term_freqs:
            return []
        scored: list[tuple[float, int]] = []
        for index, freqs in enumerate(self._term_freqs):
            length = self._lengths[index]
            norm = BM25_K1 * (
                1 - BM25_B + BM25_B * (length / self._avg_length if self._avg_length else 1)
            )
            score = 0.0
            for term in terms:
                term_freq = freqs.get(term)
                if not term_freq:
                    continue
                score += self._idf.get(term, 0.0) * (
                    term_freq * (BM25_K1 + 1) / (term_freq + norm)
                )
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [index for _, index in scored[:limit]]


def _bm25_sidecar_path(collection_name: str) -> Path:
    return _persist_dir() / f".bm25_{collection_name}.json"


def _bm25_cache_key(collection_name: str) -> str:
    return f"{_persist_dir()}:{_active_provider}:{collection_name}"


def _bm25_entries_from_chunks(chunks: list[Document]) -> list[dict]:
    entries = []
    for chunk in chunks:
        raw_text = chunk.metadata.get("raw_text") or chunk.page_content
        entries.append(
            {
                "key": _chunk_key(raw_text, chunk.metadata.get("page")),
                "raw_text": raw_text,
                "page": chunk.metadata.get("page"),
                "heading_path": chunk.metadata.get("heading_path"),
                "source": chunk.metadata.get("source"),
            }
        )
    return entries


def _write_bm25_sidecar(collection_name: str, chunks: list[Document]) -> None:
    """Persist chunk text for keyword search next to the vector collection."""
    try:
        _persist_dir().mkdir(parents=True, exist_ok=True)
        _bm25_sidecar_path(collection_name).write_text(
            json.dumps(
                {
                    "version": BM25_SIDECAR_VERSION,
                    "collection": collection_name,
                    "entries": _bm25_entries_from_chunks(chunks),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        log.warning("Could not write BM25 sidecar for %s", collection_name)
        return
    _BM25_CACHE.pop(_bm25_cache_key(collection_name), None)


def _load_bm25_entries(
    collection_name: str, vectorstore: Chroma
) -> Optional[list[dict]]:
    path = _bm25_sidecar_path(collection_name)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("entries")
            if payload.get("version") == BM25_SIDECAR_VERSION and isinstance(entries, list):
                return entries or None
        except (OSError, ValueError):
            log.warning("Ignoring unreadable BM25 sidecar for %s", collection_name)

    # Indexes created before hybrid search have no sidecar: rebuild keywords
    # from the vector store's stored metadata instead of re-embedding.
    try:
        raw = vectorstore._collection.get(include=["documents", "metadatas"])  # noqa: SLF001
    except Exception:  # noqa: BLE001 - hybrid search is optional
        return None
    entries = []
    for text, metadata in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        metadata = metadata or {}
        raw_text = metadata.get("raw_text") or text or ""
        entries.append(
            {
                "key": _chunk_key(raw_text, metadata.get("page")),
                "raw_text": raw_text,
                "page": metadata.get("page"),
                "heading_path": metadata.get("heading_path"),
                "source": metadata.get("source"),
            }
        )
    return entries or None


def _bm25_index_for(
    collection_name: str, vectorstore: Chroma
) -> tuple[Optional[list[dict]], Optional["_BM25Index"]]:
    entries = _load_bm25_entries(collection_name, vectorstore)
    if not entries:
        return None, None
    cache_key = _bm25_cache_key(collection_name)
    cached = _BM25_CACHE.get(cache_key)
    if cached and cached[0] == len(entries):
        return cached[1], cached[2]
    index = _BM25Index(entries)
    _BM25_CACHE[cache_key] = (len(entries), entries, index)
    return entries, index


def _reciprocal_rank_fusion(rankings: list[list[str]], limit: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + position)
    return [
        key
        for key, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :limit
        ]
    ]


def _rerank_documents(
    query: str, documents: list[Document], top_n: int
) -> list[Document]:
    """Reorder fused candidates with Jina's cross-encoder reranker.

    Uses the same JINA_API_KEY as embeddings. Any missing key, disabled
    setting, or API failure keeps the fusion order rather than failing the
    request.
    """
    candidates = documents[:RERANK_MAX_CANDIDATES]
    if not RERANK_ENABLED or not JINA_API_KEY or len(candidates) <= 1 or top_n <= 0:
        return candidates[:top_n]
    texts = [
        (document.metadata.get("raw_text") or document.page_content)[:4000]
        for document in candidates
    ]
    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": texts,
                "top_n": min(top_n, len(texts)),
                "return_documents": False,
            },
            timeout=RERANK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("Reranking unavailable (%s); keeping fusion order.", exc)
        return candidates[:top_n]

    ordered: list[Document] = []
    seen: set[int] = set()
    for result in sorted(
        results, key=lambda item: item.get("relevance_score", 0.0), reverse=True
    ):
        index = result.get("index")
        if isinstance(index, int) and 0 <= index < len(candidates) and index not in seen:
            seen.add(index)
            ordered.append(candidates[index])
    return ordered[:top_n] or candidates[:top_n]


def _retrieve_once(query: str, collection_name: str, k: int) -> str:
    """Hybrid retrieval: vector search + BM25, fused with RRF, then reranked.

    The vector arm keeps MMR diversity over a wider candidate set; BM25 adds
    exact keyword/name/number matching that embeddings often miss. RRF merges
    both rankings, then the cross-encoder reranker chooses the final `k`.
    """
    vectorstore = _get_vectorstore(collection_name)
    candidate_k = max(k * MMR_FETCH_MULTIPLIER, k)
    try:
        vector_docs = vectorstore.max_marginal_relevance_search(
            query, k=candidate_k, fetch_k=candidate_k * 2
        )
    except AttributeError:  # pragma: no cover - langchain_chroma has MMR
        vector_docs = vectorstore.similarity_search(query, k=candidate_k)

    by_key: dict[str, Document] = {_document_key(doc): doc for doc in vector_docs}
    rankings = [[_document_key(doc) for doc in vector_docs]]

    entries, bm25 = _bm25_index_for(collection_name, vectorstore)
    if entries and bm25 is not None:
        keyword_hits = bm25.search(query, limit=candidate_k)
        rankings.append([entries[index]["key"] for index in keyword_hits])
        for index in keyword_hits:
            entry = entries[index]
            key = entry["key"]
            if key in by_key:
                continue
            by_key[key] = Document(
                page_content=entry.get("raw_text") or "",
                metadata={
                    "page": entry.get("page"),
                    "heading_path": entry.get("heading_path"),
                    "source": entry.get("source"),
                    "raw_text": entry.get("raw_text") or "",
                },
            )

    fused_keys = _reciprocal_rank_fusion(rankings, limit=candidate_k)
    documents = [by_key[key] for key in fused_keys if key in by_key]
    if not documents:
        documents = vector_docs
    return _format_results(_rerank_documents(query, documents, k))


def _reindex_collection(collection_name: str) -> bool:
    """Rebuild just one corrupt collection from its stored source PDF.

    A full `client.reset()` here would silently delete every other indexed
    document, so recovery is scoped to the collection that failed.
    """
    meta_path = _collection_meta_path(collection_name)
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    source = Path(str(meta.get("source", "")))
    if not source.is_file():
        return False
    try:
        index_pdf(source, name=meta.get("name"))
        return True
    except Exception:  # noqa: BLE001 - surface the original retrieval error
        log.exception("Could not re-index corrupt collection %s.", collection_name)
        return False


def _retrieve_with_rebuild(query: str, collection_name: str, k: int) -> str:
    """Retrieve, re-indexing only the failing collection when it is corrupt."""
    try:
        return _retrieve_once(query, collection_name, k)
    except Exception as exc:  # noqa: BLE001
        if not _is_persist_corruption_error(exc):
            raise
        log.warning(
            "Collection %s unusable during retrieval (%s); re-indexing it.",
            collection_name, exc,
        )
        if not _reindex_collection(collection_name):
            raise
        return _retrieve_once(query, collection_name, k)


# ---- Uploaded-PDF support ----------------------------------------------------
# Each uploaded PDF lives in its own hash-derived collection; at most one is
# "active" at a time and the LLM's `get_rag_chunks` tool queries that one.
# Nothing is indexed until the user uploads a PDF.

DOC_PREFIX = "rag_"
_ACTIVE_COLLECTION: Optional[str] = None


def collection_name_for_sha(pdf_sha256: str) -> str:
    """Stable Chroma collection name derived from a PDF's content hash."""
    return f"{DOC_PREFIX}{pdf_sha256[:16]}"


def _collection_marker_path(collection_name: str) -> Path:
    return _persist_dir() / f".marker_{collection_name}"


def _collection_meta_path(collection_name: str) -> Path:
    return _persist_dir() / f".meta_{collection_name}.json"


def _collection_is_current(
    vectorstore: Chroma, collection_name: str, pdf_sha256: str, expected_count: int
) -> bool:
    """True when the collection holds the expected chunks for this hash."""
    try:
        count = vectorstore._collection.count()  # noqa: SLF001 - no public helper
    except Exception:  # noqa: BLE001
        count = 0
    if count != expected_count:
        return False
    marker = _collection_marker_path(collection_name)
    if not marker.is_file():
        return False
    return marker.read_text(encoding="utf-8").strip() == _expected_marker(pdf_sha256)


def _contextualize_chunk(chunk: Document, doc_name: str) -> Document:
    """Embed a chunk with its document/section context, display the raw text.

    Contextual retrieval (Anthropic-style): the embedded text is prefixed
    with ``[document > section]`` so semantically related passages that share
    no words still co-locate; the raw passage is kept in metadata so retrieval
    output stays clean and quotable.
    """
    context_parts = []
    if doc_name:
        context_parts.append(doc_name)
    heading_path = chunk.metadata.get("heading_path")
    if heading_path:
        context_parts.append(heading_path)
    raw_text = chunk.page_content.strip()
    chunk.metadata["raw_text"] = raw_text
    chunk.metadata["ctx_doc"] = doc_name
    if context_parts:
        chunk.page_content = f"[{' > '.join(context_parts)}]\n{raw_text}"
    return chunk


def _write_chunks(
    vectorstore: Chroma,
    chunks: list[Document],
    progress: Optional[Callable[[int, int], None]] = None,
    doc_name: str = "",
) -> None:
    """Add all chunks in batches, reporting optional progress."""
    chunks = [_contextualize_chunk(chunk, doc_name) for chunk in chunks]
    total = len(chunks)
    batch_size = _embed_batch_size()
    for start in range(0, total, batch_size):
        _add_documents_with_retries(vectorstore, chunks[start : start + batch_size])
        time.sleep(EMBED_BATCH_QUIET_SECONDS)
        if progress:
            progress(min(start + batch_size, total), total)


def _prepare_collection(vectorstore: Chroma, collection_name: str) -> Chroma:
    """Delete and recreate a collection so re-indexing starts from scratch."""
    try:
        vectorstore.delete_collection()
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    _BM25_CACHE.pop(_bm25_cache_key(collection_name), None)
    return _get_vectorstore(collection_name)


def index_pdf(
    pdf_path: Path | str,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    name: Optional[str] = None,
) -> int:
    """Index a PDF into its own hash-derived collection; returns chunk count.

    `progress(done, total)` (done/total in chunks) is invoked once per
    embedding batch so the UI can render a meaningful progress bar. Re-indexing
    is skipped entirely when the file was already indexed unchanged. Corrupt
    stores are wiped and rebuilt automatically.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found at {path}.")
    pages = _load_pdf_pages_from(path)
    if not pages:
        raise ValueError(
            f"PDF at {path} contains no extractable text. "
            "It may be a scanned/image-only PDF with no text layer."
        )

    sha = _pdf_hash(path)
    collection_name = collection_name_for_sha(sha)
    chunks = _split_documents(pages)
    vectorstore = _get_vectorstore(collection_name)
    if _collection_is_current(vectorstore, collection_name, sha, len(chunks)):
        _write_bm25_sidecar(collection_name, chunks)
        return len(chunks)

    try:
        _write_chunks(
            _prepare_collection(vectorstore, collection_name),
            chunks,
            progress=progress,
            doc_name=name or path.name,
        )
    except Exception as exc:  # noqa: BLE001
        if not _is_persist_corruption_error(exc):
            raise
        log.warning("Store broke while indexing %s (%s); wiping and retrying.", path.name, exc)
        _rebuild_store()
        _write_chunks(
            _prepare_collection(_get_vectorstore(collection_name), collection_name),
            chunks,
            progress=progress,
            doc_name=name or path.name,
        )

    _persist_dir().mkdir(parents=True, exist_ok=True)
    _collection_marker_path(collection_name).write_text(
        _expected_marker(sha), encoding="utf-8"
    )
    total = len(chunks)
    _collection_meta_path(collection_name).write_text(
        json.dumps(
            {
                "collection": collection_name,
                "name": name or path.name,
                "source": str(path),
                "sha256": sha,
                "embedding_provider": _active_provider,
                "embedding_model": GOOGLE_EMBEDDING_MODEL if _active_provider == "google" else JINA_EMBEDDING_MODEL,
                "pages": len(pages),
                "chunks": total,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_bm25_sidecar(collection_name, chunks)
    return total


def _active_collection_path() -> Path:
    return _persist_dir() / ".active_collection"


def set_active_collection(collection_name: Optional[str]) -> None:
    """Choose which indexed document `retrieve_active_chunks` queries.

    The selection is persisted next to the vector store so it survives app
    restarts (the in-memory value alone resets on every new process).
    """
    global _ACTIVE_COLLECTION
    _ACTIVE_COLLECTION = collection_name or None
    if _ACTIVE_COLLECTION:
        # An indexed document and a raw document must never both be active:
        # retrieval would otherwise answer from the wrong file.
        set_active_raw(None)
    try:
        _persist_dir().mkdir(parents=True, exist_ok=True)
        path = _active_collection_path()
        if _ACTIVE_COLLECTION:
            path.write_text(_ACTIVE_COLLECTION, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    except OSError:  # noqa: S110 - persistence is best-effort
        pass


def active_collection() -> Optional[str]:
    """Return the currently selected collection, or None when unset."""
    global _ACTIVE_COLLECTION
    if _ACTIVE_COLLECTION is None:
        path = _active_collection_path()
        if path.is_file():
            try:
                name = path.read_text(encoding="utf-8").strip()
                if name and _collection_meta_path(name).is_file():
                    _ACTIVE_COLLECTION = name
            except OSError:  # noqa: S110 - corrupt sidecar, treat as unset
                pass
    return _ACTIVE_COLLECTION


def active_source_label() -> Optional[str]:
    """SystemMessage-friendly description of the active document.

    Returns None while no document is active so the LLM is not nudged by
    extra context it does not need.
    """
    collection_name = active_collection()
    if not collection_name:
        return None
    label = collection_name
    meta = _collection_meta_path(collection_name)
    if meta.is_file():
        try:
            label = json.loads(meta.read_text(encoding="utf-8")).get("name") or label
        except ValueError:  # noqa: S110 - corrupt sidecar, keep collection id
            pass
    return (
        "The user's active document in the RAG retriever is "
        f"{label!r} (indexed from an uploaded PDF). "
        "Use the get_rag_chunks tool to answer questions about its contents "
        "and answer strictly from the retrieved chunks."
    )


def list_indexed_documents() -> list[dict]:
    """Return metadata records for every uploaded indexed PDF."""
    docs: list[dict] = []
    for meta_file in _persist_dir().glob(f".meta_{DOC_PREFIX}*.json"):
        try:
            docs.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except ValueError:
            continue
    docs.sort(key=lambda d: d.get("indexed_at", ""), reverse=True)
    return docs


def remove_indexed_document(collection_name: str) -> None:
    """Delete an uploaded document's collection, markers, and metadata."""
    try:
        _get_vectorstore(collection_name).delete_collection()
    except Exception:  # noqa: BLE001
        log.warning("Could not delete collection %s", collection_name)
    _BM25_CACHE.pop(_bm25_cache_key(collection_name), None)
    for path in (
        _collection_marker_path(collection_name),
        _collection_meta_path(collection_name),
        _bm25_sidecar_path(collection_name),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    if _ACTIVE_COLLECTION == collection_name:
        set_active_collection(None)


def retrieve_active_chunks(query: str = DEFAULT_QUERY, k: int = DEFAULT_TOP_K) -> str:
    """Retrieve chunks from the active user-uploaded document.

    Returns a prompt for the LLM when no document has been uploaded yet,
    rather than querying an empty store. When a raw (un-embedded) document is
    active, returns its full text instead of searching.
    """
    if active_raw():
        full_text = retrieve_full_text()
        if full_text:
            return (
                "The active document is in raw mode (RAG disabled) — here is "
                "its full text; answer strictly from it.\n\n" + full_text
            )
    collection_name = active_collection()
    if not collection_name:
        return (
            "No document has been uploaded to the RAG retriever yet. "
            "Ask the user to upload a PDF before using this tool."
        )
    try:
        return _retrieve_with_rebuild(query, collection_name, k)
    except Exception as exc:  # noqa: BLE001
        log.warning("RAG retrieval failed: %s", exc)
        return (
            "RAG retrieval is temporarily unavailable "
            f"(external API error: {exc})."
        )


# ---- Raw (no-embedding) PDF support ------------------------------------------
# When RAG is switched off the app never chunks or embeds anything: the
# uploaded PDF is kept as-is and its FULL text is handed to the model as
# context. A raw document is tracked by its own sidecar files and never
# touches Chroma or the embedding API.

RAW_PREFIX = "raw_"
_ACTIVE_RAW: Optional[str] = None


def raw_id_for_sha(pdf_sha256: str) -> str:
    """Stable raw-document id derived from a PDF's content hash."""
    return f"{RAW_PREFIX}{pdf_sha256[:16]}"


def _raw_meta_path(raw_id: str) -> Path:
    return _persist_dir() / f".meta_{raw_id}.json"


def _raw_active_path() -> Path:
    return _persist_dir() / ".active_raw_document"


def set_active_raw(raw_id: Optional[str]) -> None:
    """Choose which stored PDF is the raw (un-embedded) active document.

    The selection is persisted next to the vector store so it survives app
    restarts (the in-memory value alone resets on every new process).
    """
    global _ACTIVE_RAW
    _ACTIVE_RAW = raw_id or None
    if _ACTIVE_RAW:
        # Selecting a raw document supersedes any indexed selection.
        set_active_collection(None)
    try:
        _persist_dir().mkdir(parents=True, exist_ok=True)
        path = _raw_active_path()
        if _ACTIVE_RAW:
            path.write_text(_ACTIVE_RAW, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    except OSError:  # noqa: S110 - persistence is best-effort
        pass


def active_raw() -> Optional[str]:
    """Return the currently selected raw document id, or None when unset."""
    global _ACTIVE_RAW
    if _ACTIVE_RAW is None:
        path = _raw_active_path()
        if path.is_file():
            try:
                name = path.read_text(encoding="utf-8").strip()
                if name and _raw_meta_path(name).is_file():
                    _ACTIVE_RAW = name
            except OSError:  # noqa: S110 - corrupt sidecar, treat as unset
                pass
    return _ACTIVE_RAW


def raw_meta(raw_id: str) -> Optional[dict]:
    """Metadata for a raw document, or None when the sidecar is missing."""
    meta_path = _raw_meta_path(raw_id)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:  # noqa: S110 - corrupt sidecar
        return None


def store_raw_pdf(
    pdf_path: Path | str,
    *,
    name: Optional[str] = None,
) -> dict:
    """Save a PDF as the active raw document WITHOUT chunking or embedding.

    Only the file on disk and a small sidecar are recorded — no Chroma
    collection is touched, so no embedding API calls are made. Returns the
    metadata record.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found at {path}.")
    pages = _load_pdf_pages_from(path)
    if not pages:
        raise ValueError(
            f"PDF at {path} contains no extractable text. "
            "It may be a scanned/image-only PDF with no text layer."
        )
    sha = _pdf_hash(path)
    raw_id = raw_id_for_sha(sha)
    meta = {
        "collection": raw_id,
        "name": name or path.name,
        "source": str(path),
        "sha256": sha,
        "embedding_model": None,
        "pages": len(pages),
        "chunks": 0,
        "raw": True,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _persist_dir().mkdir(parents=True, exist_ok=True)
    _raw_meta_path(raw_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    set_active_raw(raw_id)
    return meta


def remove_raw_document(raw_id: Optional[str]) -> None:
    """Delete a raw document's sidecar and clear it when it is active."""
    if not raw_id:
        return
    try:
        _raw_meta_path(raw_id).unlink(missing_ok=True)
    except OSError:
        pass
    if active_raw() == raw_id:
        set_active_raw(None)


def retrieve_full_text() -> Optional[str]:
    """The COMPLETE text of the active document, page by page.

    Used when RAG is switched off: nothing is chunked, embedded, or retrieved
    — the model simply receives the whole document as context. Falls back to
    the currently selected indexed document's source PDF when no raw document
    has been stored, so toggling RAG off mid-session still has full text.
    """
    source: Optional[Path] = None
    raw_id = active_raw()
    if raw_id:
        meta = raw_meta(raw_id)
        if meta:
            source = Path(meta.get("source", ""))
    if source is None or not source.is_file():
        collection_name = active_collection()
        if collection_name:
            meta_path = _collection_meta_path(collection_name)
            if meta_path.is_file():
                try:
                    source = Path(
                        json.loads(meta_path.read_text(encoding="utf-8")).get(
                            "source", ""
                        )
                    )
                except ValueError:  # noqa: S110 - corrupt sidecar
                    source = None
    if source is None or not source.is_file():
        return None
    pages = _load_pdf_pages_from(source)
    if not pages:
        return None
    parts: list[str] = []
    for document in pages:
        page_num = document.metadata.get("page", "?")
        if isinstance(page_num, int):
            page_num = page_num + 1
        parts.append(f"--- Page {page_num} ---\n{document.page_content.strip()}")
    return "\n\n".join(parts)
