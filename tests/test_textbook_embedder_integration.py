"""Chạy riêng với E5 thật bằng cách đặt RUN_EMBEDDING_INTEGRATION=1.

Model cần có sẵn trong cache hoặc được tải ở lần chạy đầu tiên.
Đặt EMBEDDING_MODEL_CACHE và EMBEDDING_LOCAL_FILES_ONLY=1 để kiểm tra offline.
"""

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1",
    reason="Đặt RUN_EMBEDDING_INTEGRATION=1 để chạy model E5 thật",
)


def test_real_e5_documents_and_query():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "textbook_embedder_integration", root / "src/ingestion/textbook_embedder.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    embedder = module.E5Embedder(
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3", batch_size=2,
        cache_folder=os.environ.get("EMBEDDING_MODEL_CACHE"),
        local_files_only=os.environ.get("EMBEDDING_LOCAL_FILES_ONLY") == "1",
    )
    chunks = [
        Document(page_content="A database is a collection of related data.",
                 metadata={"sources": [{"record_id": "p0:b1"}]}),
        Document(page_content="Một giao dịch gồm nhiều thao tác với cơ sở dữ liệu.",
                 metadata={"sources": [{"record_id": "p0:b2"}]}),
    ]
    original = deepcopy(chunks)

    vectors = embedder.embed_documents(chunks)
    query = embedder.embed_query("Cơ sở dữ liệu là gì?")

    assert vectors.shape == (2, 384)
    assert query.shape == (384,)
    for values in (vectors, query):
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-5)
    assert chunks == original
    # Embed riêng từng chunk để kiểm tra kết quả vẫn khớp thứ tự trong batch.
    for index, chunk in enumerate(chunks):
        single = embedder.embed_documents([chunk])[0]
        np.testing.assert_allclose(vectors[index], single, atol=1e-5)
