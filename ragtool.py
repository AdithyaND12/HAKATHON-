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
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pymupdf
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
JINA_EMBEDDING_MODEL = config.JINA_EMBEDDING_MODEL
EMBEDDING_PROVIDER = config.EMBEDDING_PROVIDER
GOOGLE_EMBEDDING_MODEL = config.GOOGLE_EMBEDDING_MODEL
HTTP_TIMEOUT_SECONDS = config.HTTP_TIMEOUT_SECONDS
JINA_EMBED_BATCH_SIZE = config.JINA_EMBEDDING_BATCH_SIZE
GOOGLE_EMBED_BATCH_SIZE = config.GOOGLE_EMBEDDING_BATCH_SIZE


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
    if EMBEDDING_PROVIDER == "google":
        return _model_slug(GOOGLE_EMBEDDING_MODEL)
    return _model_slug(JINA_EMBEDDING_MODEL)

PERSIST_DIRECTORY = STORE_ROOT / _active_model_slug()
DEFAULT_QUERY = "the user's document"
DEFAULT_TOP_K = 4


def _embed_batch_size() -> int:
    """Chunks per embedding batch, provider-aware."""
    if EMBEDDING_PROVIDER == "google":
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
    if EMBEDDING_PROVIDER == "google":
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


def _get_client() -> PersistentClient:
    """One shared persistent client per process.

    Chroma's writes go through an async write queue watched by the client.
    Creating a NEW client per call means a querying client may never see the
    other client's still-pending writes, causing spurious "collection not
    initialized" failures. A single shared client keeps index + query coherent.
    """
    global _CLIENT
    if _CLIENT is None:
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        # allow_reset lets the corruption self-heal wipe the store in-process.
        _CLIENT = PersistentClient(
            path=str(PERSIST_DIRECTORY), settings=Settings(allow_reset=True)
        )
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
    log.warning("Resetting Chroma store %s and rebuilding from scratch.", PERSIST_DIRECTORY)
    _get_client().reset()


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
    model = GOOGLE_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "google" else JINA_EMBEDDING_MODEL
    return f"{EMBEDDING_PROVIDER}:{model}\n{INDEX_PIPELINE_VERSION}\n{pdf_sha256}"


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


def _retrieve_once(query: str, collection_name: str, k: int) -> str:
    """Run one retrieval and format the results.

    Uses Maximal Marginal Relevance so the k slots cover distinct passages
    instead of several near-identical hits; falls back to plain similarity
    search for stores without an MMR implementation.
    """
    vectorstore = _get_vectorstore(collection_name)
    try:
        results = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=k * MMR_FETCH_MULTIPLIER
        )
    except AttributeError:  # pragma: no cover - langchain_chroma has MMR
        results = vectorstore.similarity_search(query, k=k)
    return _format_results(results)


def _retrieve_with_rebuild(query: str, collection_name: str, k: int) -> str:
    """Retrieve, self-healing corrupt on-disk stores with a wipe + retry."""
    try:
        return _retrieve_once(query, collection_name, k)
    except Exception as exc:  # noqa: BLE001
        if not _is_persist_corruption_error(exc):
            raise
        log.warning("Store unusable during retrieval (%s); wiping and retrying.", exc)
        _rebuild_store()
        try:
            return _retrieve_once(query, collection_name, k)
        except Exception as retry_exc:  # noqa: BLE001
            raise retry_exc from exc


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
    return PERSIST_DIRECTORY / f".marker_{collection_name}"


def _collection_meta_path(collection_name: str) -> Path:
    return PERSIST_DIRECTORY / f".meta_{collection_name}.json"


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

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
                "embedding_provider": EMBEDDING_PROVIDER,
                "embedding_model": GOOGLE_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "google" else JINA_EMBEDDING_MODEL,
                "pages": len(pages),
                "chunks": total,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return total


def _active_collection_path() -> Path:
    return PERSIST_DIRECTORY / ".active_collection"


def set_active_collection(collection_name: Optional[str]) -> None:
    """Choose which indexed document `retrieve_active_chunks` queries.

    The selection is persisted next to the vector store so it survives app
    restarts (the in-memory value alone resets on every new process).
    """
    global _ACTIVE_COLLECTION
    _ACTIVE_COLLECTION = collection_name or None
    try:
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
    collection_name = _ACTIVE_COLLECTION
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
    for meta_file in PERSIST_DIRECTORY.glob(f".meta_{DOC_PREFIX}*.json"):
        try:
            docs.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except ValueError:
            continue
    docs.sort(key=lambda d: d.get("indexed_at", ""), reverse=True)
    return docs


def remove_indexed_document(collection_name: str) -> None:
    """Delete an uploaded document's collection, markers, and metadata."""
    global _ACTIVE_COLLECTION
    try:
        _get_vectorstore(collection_name).delete_collection()
    except Exception:  # noqa: BLE001
        log.warning("Could not delete collection %s", collection_name)
    for path in (
        _collection_marker_path(collection_name),
        _collection_meta_path(collection_name),
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
    collection_name = _ACTIVE_COLLECTION
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
    return PERSIST_DIRECTORY / f".meta_{raw_id}.json"


def _raw_active_path() -> Path:
    return PERSIST_DIRECTORY / ".active_raw_document"


def set_active_raw(raw_id: Optional[str]) -> None:
    """Choose which stored PDF is the raw (un-embedded) active document.

    The selection is persisted next to the vector store so it survives app
    restarts (the in-memory value alone resets on every new process).
    """
    global _ACTIVE_RAW
    _ACTIVE_RAW = raw_id or None
    try:
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
