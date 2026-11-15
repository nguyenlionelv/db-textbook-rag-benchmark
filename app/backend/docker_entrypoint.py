"""Adapt the saved manifest to container paths and start the web application."""

import json
import os
from pathlib import Path


def prepare_manifest(source, qdrant_path, destination):
    """Write a runtime copy, leaving the original ingestion manifest unchanged."""
    profile = json.loads(Path(source).read_text(encoding="utf-8"))
    profile["qdrant_path"] = str(qdrant_path)
    Path(destination).write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    return str(destination)


def main():
    """Use the mounted index and keep the application on a single worker."""
    if os.environ.get("RAG_MODE", "live") != "examples":
        os.environ["RAG_INDEX_MANIFEST"] = prepare_manifest(
            os.environ.get("RAG_INDEX_MANIFEST", "/data/index_manifest.json"),
            os.environ.get("RAG_QDRANT_PATH", "/data/qdrant"),
            "/tmp/rag_index_manifest.json",
        )
    os.execvp("python", ["python", "-m", "uvicorn", "app.backend.main:app",
                         "--host", "0.0.0.0", "--port", "8000", "--workers", "1"])


if __name__ == "__main__":
    main()
