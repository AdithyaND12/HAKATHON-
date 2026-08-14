import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

import ragtool


def test_no_default_pdf_is_configured():
    assert not hasattr(ragtool, "PDF_PATH")
    assert not hasattr(ragtool, "DEFAULT_PDF_PATH")


def test_retrieve_without_active_document_prompts_for_upload(monkeypatch):
    ragtool.set_active_collection(None)
    out = ragtool.retrieve_active_chunks("anything")
    assert "No document has been uploaded" in out


def test_embedding_model_uses_environment_configuration(monkeypatch):
    monkeypatch.setattr(ragtool, "JINA_EMBEDDING_MODEL", "jina-test-model")
    monkeypatch.setattr(ragtool, "JINA_API_KEY", "test-key")

    embedding_model = ragtool._get_embedding_model()

    assert embedding_model.model_name == "jina-test-model"


# ---- Uploaded-PDF support ----------------------------------------------------


class _CountCollection:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


class _FakeVectorstore:
    def __init__(self, count=0):
        self._collection = _CountCollection(count)
        self.added = []
        self.deleted = False

    def add_documents(self, documents):
        self.added.extend(documents)

    def similarity_search(self, query, k=4):
        return [Document(page_content="the clause about speech", metadata={"page": 3})]

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20):
        return self.similarity_search(query, k=k)

    def delete_collection(self):
        self.deleted = True


def test_collection_name_is_hash_derived():
    assert ragtool.collection_name_for_sha("a" * 64) == "rag_" + "a" * 16


def test_model_slug_is_filesystem_safe():
    assert ragtool._model_slug("jina-embeddings-v4") == "jina-embeddings-v4"
    assert ragtool._model_slug("Weird/Model NAME 2.0") == "weird-model-name-2.0"
    assert ragtool._model_slug("!!!") == "default"


def test_persist_directory_is_namespaced_by_model(monkeypatch):
    monkeypatch.setattr(ragtool, "JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
    import importlib
    importlib.reload(ragtool)
    assert "chroma_stores" in str(ragtool.PERSIST_DIRECTORY)
    assert "jina-embeddings-v4" in str(ragtool.PERSIST_DIRECTORY)


def test_persist_corruption_error_detection():
    broken = "Chroma collection not initialized. Use `reset_collection` to recreate."
    assert ragtool._is_persist_corruption_error(Exception(broken))
    assert ragtool._is_persist_corruption_error(
        Exception("InvalidDimensionException: dimension mismatch (1024 vs 3072)")
    )
    assert not ragtool._is_persist_corruption_error(
        Exception("429 rate limit exceeded, retry in 5s")
    )


def test_retrieve_self_heals_corrupt_store(monkeypatch, tmp_path):
    """A store that raises 'not initialized' on first use must be wiped and
    rebuilt instead of surfacing the error to the caller."""
    from langchain_core.documents import Document

    calls = {"n": 0}
    results = [Document(page_content="recovered chunk", metadata={"page": 2})]

    def fake_retrieve_once(query, collection_name, k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception(
                "Chroma collection not initialized. Use `reset_collection` to recreate."
            )
        return ragtool._format_results(results)

    monkeypatch.setattr(ragtool, "_retrieve_once", fake_retrieve_once)
    monkeypatch.setattr(ragtool, "PERSIST_DIRECTORY", tmp_path)
    monkeypatch.setattr(ragtool, "_rebuild_store", lambda: None)

    out = ragtool._retrieve_with_rebuild("speech", ragtool.collection_name_for_sha("d" * 64), 4)

    assert calls["n"] == 2
    assert "recovered chunk" in out


def test_index_pdf_builds_collection_and_writes_sidecar(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfake bytes\n")
    sha = ragtool._pdf_hash(pdf)
    collection = ragtool.collection_name_for_sha(sha)

    store = _FakeVectorstore()
    monkeypatch.setattr(ragtool, "_get_vectorstore", lambda collection_name: store)
    monkeypatch.setattr(ragtool, "PERSIST_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        ragtool,
        "_load_pdf_pages_from",
        lambda path: [
            Document(page_content="hello document text", metadata={"source": str(path), "page": 0})
        ],
    )
    monkeypatch.setattr(ragtool, "_split_documents", lambda docs: docs)
    monkeypatch.setattr(ragtool.time, "sleep", lambda seconds: None)

    progress_seen = []
    count = ragtool.index_pdf(pdf, progress=lambda done, total: progress_seen.append((done, total)), name="doc.pdf")

    assert count == 1
    assert len(store.added) == 1
    assert progress_seen == [(1, 1)]
    meta = json.loads((tmp_path / f".meta_{collection}.json").read_text(encoding="utf-8"))
    assert meta["name"] == "doc.pdf"
    assert meta["chunks"] == 1
    assert meta["embedding_model"] == ragtool.JINA_EMBEDDING_MODEL
    assert (tmp_path / f".marker_{collection}").read_text(encoding="utf-8") == ragtool._expected_marker(sha)


def test_index_pdf_skips_reindexing_when_unchanged(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfake bytes\n")
    sha = ragtool._pdf_hash(pdf)
    collection = ragtool.collection_name_for_sha(sha)
    (tmp_path / f".marker_{collection}").write_text(
        ragtool._expected_marker(sha), encoding="utf-8"
    )

    store = _FakeVectorstore(count=1)
    monkeypatch.setattr(
        ragtool,
        "_get_vectorstore",
        lambda collection_name=collection: store,
    )
    monkeypatch.setattr(ragtool, "PERSIST_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        ragtool,
        "_load_pdf_pages_from",
        lambda path: [Document(page_content="x", metadata={"page": 0})],
    )

    progress_seen = []
    count = ragtool.index_pdf(pdf, progress=lambda done, total: progress_seen.append((done, total)))

    assert count == 1
    assert store.deleted is False
    assert store.added == []
    assert progress_seen == []


def test_index_pdf_rebuilds_when_embedding_model_changed(monkeypatch, tmp_path):
    """Vectors from a different embedding model are incompatible: an old-style
    marker (e.g. written by the previous Gemini embedding setup) must trigger a
    full rebuild instead of a silent skip."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfake bytes\n")
    sha = ragtool._pdf_hash(pdf)
    collection = ragtool.collection_name_for_sha(sha)
    (tmp_path / f".marker_{collection}").write_text(
        f"gemini-embedding-001\n{sha}", encoding="utf-8"
    )

    store = _FakeVectorstore(count=1)
    monkeypatch.setattr(
        ragtool,
        "_get_vectorstore",
        lambda collection_name=collection: store,
    )
    monkeypatch.setattr(ragtool, "PERSIST_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        ragtool,
        "_load_pdf_pages_from",
        lambda path: [Document(page_content="x", metadata={"page": 0})],
    )
    monkeypatch.setattr(ragtool, "_split_documents", lambda docs: docs)
    monkeypatch.setattr(ragtool.time, "sleep", lambda seconds: None)

    count = ragtool.index_pdf(pdf)

    assert count == 1
    assert store.deleted is True
    assert len(store.added) == 1


def test_index_pdf_rejects_empty_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ragtool, "_load_pdf_pages_from", lambda path: [])

    with pytest.raises(ValueError, match="no extractable text"):
        ragtool.index_pdf(pdf)


def test_retrieve_active_chunks_queries_uploaded_collection(monkeypatch):
    seen = []

    class _Store:
        def __init__(self, collection_name):
            seen.append(collection_name)

        def max_marginal_relevance_search(self, query, k=4, fetch_k=20):
            return [Document(page_content="the clause about speech", metadata={"page": 3})]

    monkeypatch.setattr(ragtool, "_get_vectorstore", _Store)
    monkeypatch.setattr(ragtool.time, "sleep", lambda seconds: None)

    uploaded = ragtool.collection_name_for_sha("f" * 64)
    ragtool.set_active_collection(uploaded)
    try:
        out = ragtool.retrieve_active_chunks("speech")
    finally:
        ragtool.set_active_collection(None)
    assert seen == [uploaded]
    assert "Page: 4" in out
    assert "the clause about speech" in out


def test_active_collection_persists_across_restarts(monkeypatch, tmp_path):
    monkeypatch.setattr(ragtool, "PERSIST_DIRECTORY", tmp_path)
    collection = ragtool.collection_name_for_sha("b" * 64)
    (tmp_path / f".meta_{collection}.json").write_text(
        json.dumps({"collection": collection, "name": "doc.pdf"}), encoding="utf-8"
    )

    ragtool.set_active_collection(collection)
    assert (tmp_path / ".active_collection").read_text(encoding="utf-8") == collection

    monkeypatch.setattr(ragtool, "_ACTIVE_COLLECTION", None)
    assert ragtool.active_collection() == collection

    ragtool.set_active_collection(None)
    assert ragtool.active_collection() is None
    assert not (tmp_path / ".active_collection").exists()


def test_active_source_label_describes_uploaded_document(monkeypatch, tmp_path):
    monkeypatch.setattr(ragtool, "PERSIST_DIRECTORY", tmp_path)
    collection = ragtool.collection_name_for_sha("e" * 64)
    (tmp_path / f".meta_{collection}.json").write_text(
        json.dumps({"collection": collection, "name": "contract.pdf", "chunks": 5}),
        encoding="utf-8",
    )

    ragtool.set_active_collection(collection)
    try:
        label = ragtool.active_source_label()
        assert label is not None
        assert "contract.pdf" in label
        assert "get_rag_chunks" in label
    finally:
        ragtool.set_active_collection(None)
    assert ragtool.active_source_label() is None


# ---- Heading-aware chunking & contextual retrieval ---------------------------


def test_split_documents_threads_heading_path_across_pages():
    pages = [
        Document(
            page_content="page one " * 200,
            metadata={"source": "d.pdf", "page": 0, ragtool._HEADINGS_META_KEY: [(1, "Overview")]},
        ),
        Document(
            page_content="page two " * 200,
            metadata={"source": "d.pdf", "page": 1, ragtool._HEADINGS_META_KEY: [(2, "Details")]},
        ),
    ]
    chunks = ragtool._split_documents(pages)

    assert all("heading" in c.metadata for c in chunks)
    assert all(c.metadata["heading"] == "Overview" for c in chunks if c.metadata["page"] == 0)
    assert all(
        c.metadata["heading_path"] == "Overview > Details"
        for c in chunks
        if c.metadata["page"] == 1
    )
    assert all("_headings" not in c.metadata for c in chunks)


def test_split_documents_closes_deeper_headings_on_new_sibling():
    pages = [
        Document(
            page_content="page one " * 200,
            metadata={"source": "d.pdf", "page": 0, ragtool._HEADINGS_META_KEY: [(1, "Overview")]},
        ),
        Document(
            page_content="page two " * 200,
            metadata={"source": "d.pdf", "page": 1, ragtool._HEADINGS_META_KEY: [(2, "History")]},
        ),
        Document(
            page_content="page three " * 200,
            metadata={"source": "d.pdf", "page": 2, ragtool._HEADINGS_META_KEY: [(2, "Recent Work")]},
        ),
    ]
    chunks = ragtool._split_documents(pages)

    # Heading paths accumulate in document order, so early chunks only know
    # the headings seen so far, and a sibling heading closes deeper ones.
    assert all(
        c.metadata["heading_path"] == "Overview > History"
        for c in chunks
        if c.metadata["page"] == 1
    )
    assert all(
        c.metadata["heading_path"] == "Overview > Recent Work"
        for c in chunks
        if c.metadata["page"] == 2
    )


def test_split_documents_without_headings_has_no_section_metadata():
    pages = [Document(page_content="plain text " * 200, metadata={"source": "d.pdf", "page": 0})]
    chunks = ragtool._split_documents(pages)

    assert chunks
    assert all("heading" not in c.metadata for c in chunks)


def test_contextualize_chunk_prefixes_embedding_and_keeps_raw_text():
    chunk = Document(
        page_content="the actual passage",
        metadata={"heading_path": "Overview > Details", "page": 2},
    )

    out = ragtool._contextualize_chunk(chunk, doc_name="contract.pdf")

    assert out.page_content == "[contract.pdf > Overview > Details]\nthe actual passage"
    assert out.metadata["raw_text"] == "the actual passage"


def test_contextualize_chunk_without_context_keeps_text_unchanged():
    chunk = Document(page_content="no headings here", metadata={"page": 0})

    out = ragtool._contextualize_chunk(chunk, doc_name="")

    assert out.page_content == "no headings here"
    assert out.metadata["raw_text"] == "no headings here"


def test_format_results_shows_raw_text_and_section():
    chunk = Document(
        page_content="[contract.pdf > Overview]\nthe real passage",
        metadata={"raw_text": "the real passage", "heading_path": "Overview", "page": 2},
    )

    out = ragtool._format_results([chunk])

    assert "Page: 3" in out
    assert "Section: Overview" in out
    assert "[contract.pdf" not in out
    assert "the real passage" in out


def test_format_results_falls_back_to_embedded_text():
    chunk = Document(page_content="plain stored text", metadata={"page": 0})
    out = ragtool._format_results([chunk])
    assert "plain stored text" in out


def test_retrieve_once_uses_mmr_for_diversity(monkeypatch):
    class _MmrStore:
        def __init__(self, collection_name):
            self.calls = []

        def max_marginal_relevance_search(self, query, k=4, fetch_k=20):
            self.calls.append((query, k, fetch_k))
            return [Document(page_content="chunk a", metadata={"page": 1})]

    store = _MmrStore("x")
    monkeypatch.setattr(ragtool, "_get_vectorstore", lambda collection_name: store)

    out = ragtool._retrieve_once("speech", "rag_abc", 4)

    assert store.calls == [("speech", 4, 20)]
    assert "chunk a" in out


def test_marker_includes_pipeline_version(monkeypatch):
    monkeypatch.setattr(ragtool, "JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
    assert ragtool.INDEX_PIPELINE_VERSION >= 2
    assert ragtool._expected_marker("a" * 64) == (
        f"jina-embeddings-v4\n{ragtool.INDEX_PIPELINE_VERSION}\n{'a' * 64}"
    )


# ---- Embedding quota handling ------------------------------------------------


def test_daily_quota_exhaustion_is_detected():
    daily = (
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        "generativelanguage.googleapis.com/embed_content_free_tier_requests, "
        "limit: 1000, model: gemini-embedding-1.0. Please retry in 52.36s."
    )
    assert ragtool._is_daily_quota_exhausted(Exception(daily))
    # Plain rate limiting is NOT a terminal daily-quota error.
    assert not ragtool._is_daily_quota_exhausted(Exception("429 rate limit exceeded"))
    assert not ragtool._is_daily_quota_exhausted(Exception("server error 500"))


def test_add_documents_fails_fast_on_daily_quota(monkeypatch):
    attempts = {"n": 0}

    class _Store:
        def add_documents(self, documents):
            attempts["n"] += 1
            raise Exception(
                "429 RESOURCE_EXHAUSTED: Quota exceeded for metric: "
                "embed_content_free_tier_requests, limit: 1000, per day"
            )

    monkeypatch.setattr(ragtool.time, "sleep", lambda seconds: None)
    with pytest.raises(Exception, match="free_tier"):
        ragtool._add_documents_with_retries(
            _Store(), [Document(page_content="x", metadata={"page": 0})]
        )
    assert attempts["n"] == 1
