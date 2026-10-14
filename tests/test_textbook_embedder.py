"""Unit test dùng model giả, không tải trọng số E5."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "textbook_embedder_under_test", ROOT / "src/ingestion/textbook_embedder.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
E5Embedder = module.E5Embedder


class FakeModel:
    max_seq_length = 8

    def __init__(self, model_name, revision, device, cache_folder=None, local_files_only=False):
        self.config = (model_name, revision, device)
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def tokenizer(self, text, truncation):
        assert truncation is False
        # Mô phỏng tokenizer có thêm hai special tokens.
        return {"input_ids": [0] * (len(text.split()) + 2)}

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return np.eye(3, dtype=np.float32)[:len(inputs)]


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setattr(module, "SentenceTransformer", FakeModel)
    return E5Embedder(revision="test-revision", batch_size=2,
                      cache_folder="test-cache", local_files_only=True)


def test_loads_model_once_with_selected_configuration(embedder):
    assert embedder.model.config == (
        "intfloat/multilingual-e5-small", "test-revision", "cpu",
    )
    assert embedder.model.cache_folder == "test-cache"
    assert embedder.model.local_files_only is True
    model = embedder.model
    embedder.embed_query("database")
    embedder.embed_query("transaction")
    assert embedder.model is model
    assert len(model.calls) == 2


def test_documents_keep_order_content_and_metadata(embedder):
    chunks = [
        Document(page_content="  database systems  ", metadata={
            "sources": [{"record_id": "p0:b1", "source_start_index": 0}],
        }),
        Document(page_content="Chapter 1", metadata={"sources": []}),
    ]
    original = deepcopy(chunks)

    vectors = embedder.embed_documents(chunks)

    assert embedder.model.calls[0][0] == [
        "passage:   database systems  ", "passage: Chapter 1",
    ]
    np.testing.assert_array_equal(vectors, np.eye(3, dtype=np.float32)[:2])
    assert chunks == original


def test_query_prefix_and_one_dimensional_output(embedder):
    vector = embedder.embed_query("What is a database?")

    assert embedder.model.calls[0][0] == ["query: What is a database?"]
    assert vector.shape == (3,)
    assert vector.dtype == np.float32


def test_encode_requests_normalized_float32_without_extra_prompt(embedder):
    embedder.embed_documents([Document(page_content="database")])

    assert embedder.model.calls[0][1] == {
        "batch_size": 2,
        "precision": "float32",
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "prompt": "",
        "show_progress_bar": False,
    }


def test_empty_documents_return_empty_matrix_without_encoding(embedder):
    vectors = embedder.embed_documents([])

    assert vectors.shape == (0, 3)
    assert vectors.dtype == np.float32
    assert embedder.model.calls == []


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_document_rejects_whole_batch_before_encoding(embedder, text):
    chunks = [Document(page_content="database"), Document(page_content=text)]

    with pytest.raises(ValueError, match=r"text\[1\]"):
        embedder.embed_documents(chunks)
    assert embedder.model.calls == []


@pytest.mark.parametrize("text", ["", " \n", None, 123])
def test_invalid_query_is_rejected(embedder, text):
    with pytest.raises(ValueError, match=r"text\[0\]"):
        embedder.embed_query(text)
    assert embedder.model.calls == []


@pytest.mark.parametrize("is_query", [False, True])
def test_token_limit_counts_prefix_and_special_tokens(embedder, is_query):
    def encode(text):
        if is_query:
            return embedder.embed_query(text)
        return embedder.embed_documents([Document(page_content=text)])

    encode("one two three four five")  # 5 + prefix + 2 special tokens = 8
    with pytest.raises(ValueError, match="9 token"):
        encode("one two three four five six")
    assert len(embedder.model.calls) == 1


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_fails_before_loading_model(monkeypatch, batch_size):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Không được tải model khi cấu hình không hợp lệ")

    monkeypatch.setattr(module, "SentenceTransformer", unexpected_load)
    with pytest.raises(ValueError, match="batch_size"):
        E5Embedder(batch_size=batch_size)
