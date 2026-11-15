"""Check container path conversion without altering the saved index profile."""

import json
from pathlib import Path

import pytest

from app.backend import docker_entrypoint


def test_manifest_copy_preserves_identity_and_original(tmp_path):
    profile = {"collection": "saved", "model_revision": "pinned", "chunks": 6040,
               "qdrant_path": "D:/original/qdrant", "chunk_sha256": "original-hash"}
    original = tmp_path / "original.json"
    original.write_text(json.dumps(profile), encoding="utf-8")
    before = original.read_bytes()
    target = tmp_path / "runtime.json"
    result = docker_entrypoint.prepare_manifest(original, "/data/qdrant", target)
    assert Path(result) == target
    assert json.loads(target.read_text(encoding="utf-8")) == {**profile, "qdrant_path": "/data/qdrant"}
    assert original.read_bytes() == before


@pytest.mark.parametrize("mode", ["live", "examples"])
def test_entrypoint_prepares_live_manifest_and_executes_one_worker(monkeypatch, mode):
    calls = []
    monkeypatch.setenv("RAG_MODE", mode)
    monkeypatch.setenv("RAG_INDEX_MANIFEST", "/data/index_manifest.json")
    monkeypatch.setenv("RAG_QDRANT_PATH", "/data/qdrant")
    def prepare(*args):
        calls.append(args)
        return "/tmp/runtime.json"
    monkeypatch.setattr(docker_entrypoint, "prepare_manifest", prepare)
    monkeypatch.setattr(docker_entrypoint.os, "execvp", lambda executable, args: calls.append((executable, args)))
    docker_entrypoint.main()
    assert len(calls) == (2 if mode == "live" else 1)
    assert calls[-1][0] == "python" and calls[-1][1][-2:] == ["--workers", "1"]
