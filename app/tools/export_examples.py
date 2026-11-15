"""Export a few recorded answers without prompts, credentials, or local paths."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.backend.schemas import public_answer

EXAMPLES = {
    "f01_vi": ("Cơ sở dữ liệu là gì?", "KHÁI NIỆM NỀN TẢNG"),
    "f06_en": ("Defining a database vs. constructing it", "HIỂU VỀ DBMS"),
    "probe_ip_vi": ("Khi giáo trình chưa có đủ thông tin", "THIẾU BẰNG CHỨNG"),
}


def export_examples(run_dir, output):
    """Use completed RAG results as explicitly recorded portfolio examples."""
    state = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete", "Use a completed evaluation run"
    results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    date = re.search(r"\d{8}", run_dir.name)
    recorded_at = datetime.strptime(date.group(), "%Y%m%d").strftime("%d.%m.%Y") if date else "Ngày chưa xác định"
    rows = []
    for result in results:
        if result["question_id"] in EXAMPLES:
            title, topic = EXAMPLES[result["question_id"]]
            answer = public_answer(result, backend=config["backend"], model=config["model"],
                                   request_id="example-" + result["question_id"])
            rows.append({"title": title, "topic": topic, "recorded_at": recorded_at, "result": answer.model_dump()})
    assert len(rows) == len(EXAMPLES), "The selected recorded examples are missing"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "output/rag/local_smoke_20261114")
    parser.add_argument("--output", type=Path, default=ROOT / "app/frontend/public/examples.json")
    args = parser.parse_args()
    export_examples(args.run_dir, args.output)
