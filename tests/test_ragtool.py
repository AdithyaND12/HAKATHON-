from pathlib import Path

import pytest

import ragtool


def test_default_pdf_path_points_to_existing_pdf():
    assert ragtool.PDF_PATH == ragtool.DEFAULT_PDF_PATH
    assert ragtool.PDF_PATH.is_file()


def test_custom_pdf_path_is_read_from_environment(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom.pdf"
    monkeypatch.setenv("CONSTITUTION_PDF_PATH", str(custom_path))

    assert ragtool._configured_path() == custom_path


def test_missing_pdf_raises_actionable_error(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.pdf"
    monkeypatch.setattr(ragtool, "PDF_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="CONSTITUTION_PDF_PATH"):
        ragtool._load_pdf_pages()


def test_embedding_model_uses_environment_configuration(monkeypatch):
    monkeypatch.setattr(ragtool, "GEMINI_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setattr(ragtool, "GEMINI_API_KEY", "test-key")

    embedding_model = ragtool._get_embedding_model()

    assert embedding_model.model == "embedding-model"
