"""Minimal PDF RAG: load -> split -> embed -> retrieve -> rerank."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pymupdf
from chromadb import PersistentClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:  # pragma: no cover
    GoogleGenerativeAIEmbeddings = None

import config


GOOGLE_EMBEDDING_MODEL = config.GOOGLE_EMBEDDING_MODEL
DEFAULT_TOP_K = 4
STORE_ROOT = Path(__file__).resolve().parent / "chroma_stores"
COLLECTION_NAME = "document"
_CLIENT: Optional[PersistentClient] = None


def _model_slug(model_name: str) -> str:
    """Keep the storage path safe when tests or deployments customize it."""
    return re.sub(r"[^a-z0-9._-]+", "-", model_name.lower()).strip("-._") or "default"


PERSIST_DIRECTORY = STORE_ROOT / _model_slug(GOOGLE_EMBEDDING_MODEL)


def get_embedding_model_name() -> str:
    return GOOGLE_EMBEDDING_MODEL


def _embedding_model() -> Embeddings:
    if GoogleGenerativeAIEmbeddings is None:
        raise ImportError("Install langchain-google-genai to use PDF RAG.")
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    return GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )


def _client() -> PersistentClient:
    global _CLIENT
    if _CLIENT is None:
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        _CLIENT = PersistentClient(path=str(PERSIST_DIRECTORY))
    return _CLIENT


def _store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embedding_model(),
        client=_client(),
    )


def _load_pdf(path: Path) -> list[Document]:
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found at {path}.")

    pages = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_number},
                    )
                )
    return pages


def _split(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [
        Document(page_content=text, metadata=dict(page.metadata))
        for page in pages
        for text in splitter.split_text(page.page_content)
    ]


def _metadata_path() -> Path:
    return PERSIST_DIRECTORY / "document.json"


def _save_metadata(path: Path, name: str, pages: int, chunks: int) -> None:
    _metadata_path().write_text(
        json.dumps(
            {
                "name": name,
                "source": str(path),
                "pages": pages,
                "chunks": chunks,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _metadata() -> Optional[dict]:
    try:
        return json.loads(_metadata_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def index_pdf(
    pdf_path: Path | str,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
    name: Optional[str] = None,
) -> int:
    """Replace the current PDF with one new indexed PDF."""
    path = Path(pdf_path)
    pages = _load_pdf(path)
    if not pages:
        raise ValueError(f"PDF at {path} contains no selectable text.")

    chunks = _split(pages)
    try:
        _store().delete_collection()
    except Exception:
        pass
    _store().add_documents(chunks)
    _save_metadata(path, name or path.name, len(pages), len(chunks))

    if progress:
        progress(len(chunks), len(chunks))
    return len(chunks)


def active_source_label() -> Optional[str]:
    meta = _metadata()
    if not meta:
        return None
    return (
        f"The user's PDF is {meta['name']!r}. "
        "Use get_rag_chunks and answer strictly from the retrieved chunks."
    )


def document_metadata() -> Optional[dict]:
    return _metadata()


def _rerank(query: str, documents: list[Document], k: int) -> list[Document]:
    terms = set(re.findall(r"\w+", query.lower()))
    return sorted(
        documents,
        key=lambda document: sum(
            word in terms
            for word in re.findall(r"\w+", document.page_content.lower())
        ),
        reverse=True,
    )[:k]


def _format(documents: list[Document]) -> str:
    return "\n\n".join(
        f"--- Result {number} ---\n"
        f"Page: {document.metadata.get('page', 0) + 1}\n"
        f"{document.page_content.strip()}"
        for number, document in enumerate(documents, 1)
    )


def retrieve_active_chunks(
    query: str = "the user's document", k: int = DEFAULT_TOP_K
) -> str:
    """Retrieve candidates, rerank them, and return the final context."""
    if not _metadata_path().is_file():
        return "No PDF has been uploaded to the RAG retriever yet."

    try:
        candidates = _store().similarity_search(query, k=k * 3)
        return _format(_rerank(query, candidates, k))
    except Exception as exc:
        return f"RAG retrieval is temporarily unavailable: {exc}"
