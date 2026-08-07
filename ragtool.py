from pathlib import Path
import math
import os

import pymupdf
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_LM_STUDIO_MODEL = "qwen2.5-coder-7b-instruct"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_PDF_PATH = Path(__file__).resolve().parent / "pdfs" / "c9fe9c9b6840524844316f74bb1c556c.pdf"


def _environment_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _configured_path() -> Path:
    configured_path = Path(
        _environment_value("CONSTITUTION_PDF_PATH", str(DEFAULT_PDF_PATH))
    ).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return Path(__file__).resolve().parent / configured_path


def _configured_http_timeout() -> float:
    configured_value = os.getenv("HTTP_TIMEOUT_SECONDS")
    try:
        timeout = float(configured_value) if configured_value else DEFAULT_HTTP_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        return DEFAULT_HTTP_TIMEOUT_SECONDS
    if not math.isfinite(timeout) or timeout <= 0:
        return DEFAULT_HTTP_TIMEOUT_SECONDS
    return timeout


LM_STUDIO_MODEL = _environment_value("LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL)
LM_STUDIO_BASE_URL = _environment_value("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL)
LM_STUDIO_API_KEY = _environment_value("LM_STUDIO_API_KEY", DEFAULT_LM_STUDIO_API_KEY)
HTTP_TIMEOUT_SECONDS = _configured_http_timeout()
PDF_PATH = _configured_path()
PERSIST_DIRECTORY = Path(__file__).resolve().parent / "constitution_chroma_db"
COLLECTION_NAME = "indian_constitution"
DEFAULT_QUERY = "The Constitution of India"
DEFAULT_TOP_K = 4


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
                        metadata={
                            "source": str(PDF_PATH),
                            "page": page_number,
                        },
                    )
                )
    finally:
        pdf.close()
    return pages


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)


def _get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=LM_STUDIO_MODEL,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        chunk_size=32,
        check_embedding_ctx_length=False,
    )


def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embedding_model(),
        persist_directory=str(PERSIST_DIRECTORY),
    )


def _ensure_indexed(vectorstore: Chroma) -> None:
    if vectorstore._collection.count() > 0:
        return

    pages = _load_pdf_pages()
    chunks = _split_documents(pages)
    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[start : start + batch_size])


def retrieve_constitution_chunks(query: str = DEFAULT_QUERY) -> str:
    """Retrieve the chunks most relevant to the caller's query."""
    vectorstore = _get_vectorstore()
    _ensure_indexed(vectorstore)
    results = vectorstore.similarity_search(query, k=DEFAULT_TOP_K)

    parts: list[str] = []
    for index, document in enumerate(results, start=1):
        page_num = document.metadata.get("page", "Unknown")
        if isinstance(page_num, int):
            page_num = page_num + 1
        parts.append(
            f"--- Result {index} ---\n"
            f"Page: {page_num}\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(parts)
