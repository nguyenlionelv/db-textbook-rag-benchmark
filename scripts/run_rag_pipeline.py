"""Retrieve from a saved index and answer with a local or API language model."""

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import open_rag_pipeline, RERANKER_REVISION

CACHE = ROOT / "output/model_cache"
INDEX = ROOT / "output/ingestion/full_book/20261114_full_book_v2/index_manifest.json"
LOCAL_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def add_pipeline_arguments(parser):
    """Share runtime options between single-question and evaluation commands."""
    # Existing environment variables take priority over the local file.
    load_dotenv(ROOT / "app/.env")
    parser.add_argument("--index-manifest", type=Path, default=INDEX)
    parser.add_argument("--backend", choices=["local", "api"], default=os.getenv("RAG_BACKEND", "local"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--token-counter", help="API token counter as module:function; counts full messages")
    parser.add_argument("--api-key-env", default=os.getenv("RAG_API_KEY_ENV", "LLM_API_KEY"))
    parser.add_argument("--token-parameter", choices=["max_completion_tokens", "max_tokens"],
                        default=os.getenv("RAG_API_TOKEN_PARAMETER", "max_completion_tokens"))
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"],
                        default=os.getenv("RAG_API_REASONING_EFFORT") or None)
    parser.add_argument("--no-json-mode", action="store_true",
                        default=os.getenv("RAG_API_JSON_MODE", "true").lower() == "false")
    parser.add_argument("--num-ctx", type=int, default=int(os.getenv("RAG_NUM_CTX", "4096")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("RAG_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("RAG_TIMEOUT", "180")))
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--e5-cache", type=Path, default=CACHE / "e5_hub")
    parser.add_argument("--reranker-cache", type=Path, default=CACHE / "reranker_hub")
    parser.add_argument("--reranker-max-length", type=int, default=512)


def create_generator(args):
    """Create the selected backend without making a generation request.

    OpenAI uses its token-count endpoint by default. Other providers require
    a configured counter or matching tokenizer.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)

    common = {"num_ctx": args.num_ctx, "max_new_tokens": args.max_new_tokens, "timeout": args.timeout}
    if args.backend == "api":
        from src.generation.api_generation import create_api_generator

        args.model = args.model or os.getenv("RAG_API_MODEL")
        args.base_url = args.base_url or os.getenv("RAG_API_URL")
        args.token_counter = args.token_counter or os.getenv("RAG_API_TOKEN_COUNTER") or None
        tokenizer_path = args.tokenizer_path or os.getenv("RAG_API_TOKENIZER")
        args.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        assert args.model and args.base_url, "API requires --model and --base-url"

        return create_api_generator(
            args.model, args.base_url, token_counter=args.token_counter, tokenizer_path=args.tokenizer_path,
            api_key_env=args.api_key_env, token_parameter=args.token_parameter,
            reasoning_effort=args.reasoning_effort, json_mode=not args.no_json_mode, **common,
        )

    from transformers import AutoTokenizer
    from src.generation.textbook_generation import OllamaGenerator

    tokenizer_path = args.tokenizer_path or CACHE / "qwen3_4b_instruct_tokenizer"
    assert not args.model or args.model == LOCAL_MODEL or args.tokenizer_path, (
        "A different local model requires its matching --tokenizer-path")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    args.tokenizer_path = tokenizer_path
    return OllamaGenerator(tokenizer, model=args.model or LOCAL_MODEL,
                           base_url=args.base_url or "http://127.0.0.1:11434", **common)


def pipeline_options(args):
    """Return options accepted by the saved-index loader."""
    return {name: getattr(args, name) for name in (
        "top_k", "candidate_k", "e5_cache", "reranker_cache", "reranker_max_length")}


def run_config(args, generator):
    """Record explicit public settings; never serialize generator credentials."""
    config = {name: str(value) if isinstance(value, Path) else value
              for name, value in vars(args).items()}
    config.update(model=generator.model, base_url=generator.base_url,
                  reranker_revision=RERANKER_REVISION)
    return config


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_arguments(parser)
    parser.add_argument("--question", required=True)
    parser.add_argument("--language", choices=["vi", "en"], default="vi")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    generator = create_generator(args)
    config = run_config(args, generator)
    if args.backend == "local":
        config["local_model"] = generator.check_model()
    with open_rag_pipeline(args.index_manifest, generator, **pipeline_options(args)) as pipeline:
        result = pipeline.answer(args.question, language=args.language)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"config": config, **result}, ensure_ascii=False,
                                          indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["answer"] or result.get("error", ""))
    for source in result["citations"]:
        print(f"[{source['source_id']}] PDF pages (1-based): {[p + 1 for p in source['pdf_pages']]}")
    print(f"status={result['status']}, retrieval={result['retrieval_seconds']:.2f}s, "
          f"generation={result['generation_seconds']:.2f}s, total={result['total_seconds']:.2f}s")
    return int(result["status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
