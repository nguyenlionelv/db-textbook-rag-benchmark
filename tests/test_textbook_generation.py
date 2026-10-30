"""Test token budgets, source mapping, and Ollama responses without real models."""

from copy import deepcopy
import json

import pytest
from jsonschema import ValidationError, validate

from src.generation.textbook_generation import OllamaGenerator, build_context


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<chat>" + "\n".join(m["content"] for m in messages) + "<answer>"

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(text)


@pytest.fixture
def hits():
    return [{"point_id": 7, "page_content": "A database is a collection of related data.",
             "metadata": {"sources": [{"record_id": "book:p34:b1", "pdf_page": 34,
                                       "source_start_index": 0, "source_end_index": 43}]}}]


def test_context_preserves_input_deduplicates_and_displays_pdf_page(hits):
    original = deepcopy(hits)
    context = build_context("What is a database?", hits + hits, Tokenizer())
    assert len(context["sources"]) == 1
    assert context["sources"][0]["pdf_pages"] == [34]
    assert "[S1] PDF pages: 35" in context["context_text"]
    assert context["omitted"] == [{"point_id": 7, "reason": "duplicate"}]
    assert hits == original
    context["sources"][0]["metadata"]["sources"][0]["record_id"] = "changed"
    assert hits == original


def test_budget_drops_whole_chunks_and_keeps_later_short_chunk(hits):
    tokenizer = Tokenizer()
    full = build_context("Question", hits, tokenizer)
    long = {**hits[0], "point_id": 8, "page_content": "x" * 5000}
    context = build_context("Question", [long] + hits, tokenizer,
                            num_ctx=full["prompt_tokens"] + 20, max_new_tokens=20)
    assert context["sources"][0]["point_id"] == 7
    assert context["omitted"] == [{"point_id": 8, "reason": "token_budget"}]
    assert context["prompt_tokens"] + 20 <= full["prompt_tokens"] + 20
    assert hits[0]["page_content"] in context["context_text"]


def test_long_question_fails_before_generation():
    with pytest.raises(ValueError, match="budget"):
        build_context("q" * 5000, [], Tokenizer())


@pytest.mark.parametrize("url", ["https://ollama.com", "http://example.com", "http://localhost/api", "http://user@localhost"])
def test_rejects_nonlocal_or_ambiguous_endpoint(url):
    with pytest.raises(ValueError, match="localhost"):
        OllamaGenerator(Tokenizer(), base_url=url)


def test_empty_context_skips_ollama(monkeypatch):
    generator = OllamaGenerator(Tokenizer())
    monkeypatch.setattr(generator, "_request", lambda *args: pytest.fail("No model call expected"))
    result = generator.generate("Cơ sở dữ liệu là gì?", [])
    assert result["status"] == "insufficient_evidence" and not result["called_model"]


def response_for(generator, hits, content):
    count = build_context("Question", hits, generator.tokenizer)["prompt_tokens"]
    return {"response": json.dumps(content), "done": True, "done_reason": "stop",
            "prompt_eval_count": count, "eval_count": 20}


def test_generation_sends_raw_prompt_and_maps_only_cited_sources(hits, monkeypatch):
    generator = OllamaGenerator(Tokenizer())
    reply = response_for(generator, hits, {"status": "answered", "answer": "Related data.", "citations": ["S1"]})

    def request(path, payload):
        assert path == "/api/generate"
        assert payload["raw"] and not payload["think"] and not payload["stream"]
        assert payload["options"]["num_gpu"] == 0
        validate({"status": "answered", "answer": "Answer", "citations": ["S1"]}, payload["format"])
        validate({"status": "insufficient_evidence", "answer": "Unknown", "citations": []}, payload["format"])
        for invalid in [
            {"status": "answered", "answer": "Answer", "citations": []},
            {"status": "answered", "answer": "Answer", "citations": ["S99"]},
            {"status": "insufficient_evidence", "answer": "Unknown", "citations": ["S1"]},
        ]:
            with pytest.raises(ValidationError):
                validate(invalid, payload["format"])
        return reply

    monkeypatch.setattr(generator, "_request", request)
    result = generator.generate("Question", hits)
    assert result["status"] == "answered" and result["citation_valid"]
    assert result["citations"][0]["point_id"] == 7


@pytest.mark.parametrize("content", [
    {"status": "answered", "answer": "Answer", "citations": ["S99"]},
    {"status": "answered", "answer": "Answer", "citations": ["S1: quoted text"]},
    {"status": "answered", "answer": "Answer", "citations": []},
    {"status": "answered", "answer": "Answer", "citations": ["S1", "S1"]},
    {"status": "insufficient_evidence", "answer": "Unknown", "citations": ["S1"]},
    {"status": "answered", "answer": "", "citations": ["S1"]},
    {"answer": "No status"}, [],
])
def test_invalid_answer_is_error_not_success(hits, monkeypatch, content):
    generator = OllamaGenerator(Tokenizer())
    reply = response_for(generator, hits, content)
    monkeypatch.setattr(generator, "_request", lambda *args: reply)
    result = generator.generate("Question", hits)
    assert result["status"] == "error" and not result["citation_valid"]
    assert result["citations"] == [] and result["raw_output"]


@pytest.mark.parametrize("change", [{"done_reason": "length"}, {"prompt_eval_count": 1}, {"response": "not json"}])
def test_truncation_tokenizer_mismatch_and_invalid_json_are_errors(hits, monkeypatch, change):
    generator = OllamaGenerator(Tokenizer())
    reply = response_for(generator, hits, {"status": "answered", "answer": "Answer", "citations": ["S1"]})
    monkeypatch.setattr(generator, "_request", lambda *args: {**reply, **change})
    assert generator.generate("Question", hits)["status"] == "error"


def test_runtime_error_is_recorded(hits, monkeypatch):
    generator = OllamaGenerator(Tokenizer())

    def fail(*args):
        raise RuntimeError("server offline")

    monkeypatch.setattr(generator, "_request", fail)
    result = generator.generate("Question", hits)
    assert result["status"] == "error" and result["error"] == "server offline"
