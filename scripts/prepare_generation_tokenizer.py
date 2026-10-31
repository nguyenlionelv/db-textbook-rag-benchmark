"""Cache the tokenizer used by the Qwen3 4B Instruct experiment."""

import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
OUTPUT = ROOT / "output/model_cache/qwen3_4b_instruct_tokenizer"


def main():
    """Download the pinned tokenizer and save its source revision."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, cache_dir=ROOT / "output/model_cache/qwen3_4b_hub",
    )
    tokenizer.save_pretrained(OUTPUT)
    source = {"repo_id": MODEL, "revision": REVISION}
    (OUTPUT / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
