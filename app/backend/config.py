"""Read application settings from environment variables."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output/model_cache"


class Settings(BaseModel):
    """Server settings; keys and provider addresses stay on the server."""

    mode: Literal["live", "examples"] = "live"
    index_manifest: Path = ROOT / "output/ingestion/full_book/20261114_full_book_v2/index_manifest.json"
    e5_cache: Path = CACHE / "e5_hub"
    reranker_cache: Path = CACHE / "reranker_hub"
    local_tokenizer: Path = CACHE / "qwen3_4b_instruct_tokenizer"
    local_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_url: str = "http://127.0.0.1:11434"
    enable_local: bool = True
    api_model: str = ""
    api_url: str = ""
    api_tokenizer: Path | None = None
    api_token_counter: str = ""
    api_key_env: str = "LLM_API_KEY"
    api_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_completion_tokens"
    api_json_mode: bool = True
    api_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    num_ctx: int = Field(default=4096, gt=256)
    max_new_tokens: int = Field(default=256, gt=0)
    timeout: float = Field(default=180, gt=0)
    requests_per_hour: int = Field(default=60, ge=1)
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @classmethod
    def from_env(cls):
        """Use RAG_* variables, resolving relative paths from the repository."""
        values = {name: os.environ[f"RAG_{name.upper()}"] for name in cls.model_fields
                  if os.environ.get(f"RAG_{name.upper()}", "").strip()}
        if "allowed_origins" in values:
            values["allowed_origins"] = [origin.strip() for origin in values["allowed_origins"].split(",") if origin.strip()]
        settings = cls.model_validate(values)
        for name in ("index_manifest", "e5_cache", "reranker_cache", "local_tokenizer", "api_tokenizer"):
            path = getattr(settings, name)
            if path is not None and not path.is_absolute():
                setattr(settings, name, ROOT / path)
        return settings
