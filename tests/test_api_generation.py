"""Test API generation with fixed responses and no external requests."""

from copy import deepcopy
from http.client import IncompleteRead
from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from src.generation import api_generation
from src.generation.api_generation import APIGenerator, create_api_generator
from src.generation.textbook_generation import OllamaGenerator, build_context


def count_tokens(messages):
    """Use character counts as a predictable stand-in for provider tokenization."""
    return len(json.dumps(messages, ensure_ascii=False))


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("External API requests are not allowed in tests")

    monkeypatch.setattr(api_generation, "build_opener", unexpected_request)


@pytest.fixture
def hits():
    return [{"point_id": 7, "page_content": "A database is a collection of related data.",
             "metadata": {"sources": [{"record_id": "book:p34:b1", "pdf_page": 34,
                                       "source_start_index": 0, "source_end_index": 43}]}}]


@pytest.fixture
def generator():
    return APIGenerator("test-model", "https://provider.example/v1",
                        count_tokens=count_tokens, api_key="test-secret-key")


def response_for(answer=None, **changes):
    if answer is None:
        answer = {"status": "answered", "answer": "A collection of related data.", "citations": ["S1"]}
    response = {"id": "completion-1", "model": "test-model-snapshot",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                "usage": {"prompt_tokens": 31, "completion_tokens": 12, "total_tokens": 43}}
    return {**response, **changes}


def test_api_sends_messages_and_preserves_source_mapping(generator, hits, monkeypatch):
    original = deepcopy(hits)
    reply = response_for()

    def request(payload):
        assert payload == {"model": "test-model", "messages": prepared["messages"],
                           "stream": False, "max_completion_tokens": 256,
                           "response_format": {"type": "json_object"}}
        return reply

    prepared = build_context("What is a database?", hits, language="en", count_tokens=count_tokens)
    monkeypatch.setattr(generator, "_request", request)
    result = generator.generate("What is a database?", hits, language="en")
    assert result["status"] == "answered" and result["citation_valid"] and result["called_model"]
    assert result["citations"][0]["point_id"] == 7
    assert result["citations"][0]["pdf_pages"] == [34]
    assert result["context_text"] == prepared["context_text"]
    assert "[S1] PDF pages: 35" in result["context_text"]
    assert json.loads(result["prompt"]) == result["messages"]
    assert result["prompt_tokens"] == count_tokens(result["messages"])
    assert result["usage"] == reply["usage"]
    assert result["model"] == "test-model" and result["response_model"] == reply["model"]
    assert result["response_id"] == "completion-1" and result["seconds"] >= 0
    assert hits == original
    assert "test-secret-key" not in json.dumps(result)


@pytest.mark.parametrize("language", ["vi", "en"])
def test_empty_context_skips_api(generator, language):
    result = generator.generate("Question", [], language=language)
    assert result["status"] == "insufficient_evidence"
    assert result["citation_valid"] and not result["called_model"]
    assert result["citations"] == [] and result["usage"] is None


def test_insufficient_evidence_is_distinct_from_provider_refusal(generator, hits, monkeypatch):
    answer = {"status": "insufficient_evidence", "answer": "The excerpts do not establish this fact.", "citations": []}
    monkeypatch.setattr(generator, "_request", lambda payload: response_for(answer))
    result = generator.generate("Which university uses this database?", hits)
    assert result["status"] == "insufficient_evidence" and result["called_model"]
    assert result["citation_valid"] and result["citations"] == []


def test_multiple_citations_follow_answer_order_without_mutating_hits(generator, hits, monkeypatch):
    other = deepcopy(hits[0])
    other.update(point_id=8, page_content="A DBMS manages the database.")
    other["metadata"]["sources"][0].update(record_id="book:p35:b1", pdf_page=35)
    answer = {"status": "answered", "answer": "A DBMS manages related data.", "citations": ["S2", "S1"]}
    monkeypatch.setattr(generator, "_request", lambda payload: response_for(answer))
    result = generator.generate("Question", hits + [other])
    assert result["status"] == "answered"
    assert [citation["point_id"] for citation in result["citations"]] == [8, 7]
    result["sources"][1]["metadata"]["sources"][0]["pdf_page"] = 99
    assert other["metadata"]["sources"][0]["pdf_page"] == 35


def test_budget_skips_whole_chunks_without_renumbering_gaps(generator, hits, monkeypatch):
    short = {**hits[0], "point_id": 8, "page_content": "A transaction is a logical unit of work."}
    long = {**hits[0], "point_id": 9, "page_content": "x" * 5000}
    prepared = build_context("Question", [short], count_tokens=count_tokens)
    generator.num_ctx = prepared["prompt_tokens"] + generator.max_new_tokens
    monkeypatch.setattr(generator, "_request", lambda payload: response_for())
    result = generator.generate("Question", [long, short, short])
    assert result["status"] == "answered"
    assert [source["point_id"] for source in result["sources"]] == [8]
    assert result["sources"][0]["source_id"] == "S1"
    assert result["omitted"] == [{"point_id": 9, "reason": "token_budget"},
                                  {"point_id": 8, "reason": "duplicate"}]
    assert result["prompt_tokens"] + generator.max_new_tokens <= generator.num_ctx


def test_no_fitting_sources_skips_api(generator, hits):
    long = {**hits[0], "page_content": "x" * 5000}
    result = generator.generate("Question", [long])
    assert result["status"] == "insufficient_evidence" and not result["called_model"]
    assert result["omitted"] == [{"point_id": 7, "reason": "token_budget"}]


@pytest.mark.parametrize("query, language", [("", "vi"), ("q" * 5000, "vi"), ("Question", "fr")])
def test_invalid_input_fails_before_request(generator, query, language):
    with pytest.raises(ValueError):
        generator.generate(query, [], language=language)


def test_missing_source_is_rejected_before_request(generator, hits):
    hits[0]["metadata"]["sources"] = []
    with pytest.raises(ValueError, match="metadata.sources"):
        generator.generate("Question", hits)


@pytest.mark.parametrize("count", [None, -1, 1.5, True])
def test_invalid_counter_result_fails_before_request(generator, count):
    generator.count_tokens = lambda messages: count
    with pytest.raises(ValueError, match="count_tokens"):
        generator.generate("Question", [])


@pytest.mark.parametrize("answer", [
    {"status": "answered", "answer": "Answer", "citations": ["S99"]},
    {"status": "answered", "answer": "Answer", "citations": []},
    {"status": "answered", "answer": "Answer", "citations": ["S1", "S1"]},
    {"status": "answered", "answer": "Answer", "citations": [1]},
    {"status": "answered", "answer": " ", "citations": ["S1"]},
    {"status": "insufficient_evidence", "answer": "Unknown", "citations": ["S1"]},
    {"status": "answered", "answer": "Answer", "citations": ["S1"], "extra": True},
    {"answer": "Missing fields"}, [],
])
def test_invalid_answers_return_error(generator, hits, monkeypatch, answer):
    monkeypatch.setattr(generator, "_request", lambda payload: response_for(answer))
    result = generator.generate("Question", hits)
    assert result["status"] == "error" and not result["citation_valid"]
    assert result["answer"] == "" and result["citations"] == []
    assert result["called_model"] and result["raw_output"]


@pytest.mark.parametrize("choices", [
    None, [], [None], [{}, {}], [{}],
    [{"message": None, "finish_reason": "stop"}],
    [{"message": {"content": None, "refusal": "Cannot comply"}, "finish_reason": "stop"}],
    [{"message": {"content": "{}"}, "finish_reason": "length"}],
    [{"message": {"content": "{}"}, "finish_reason": "content_filter"}],
    [{"message": {"content": "{}", "tool_calls": [{}]}, "finish_reason": "tool_calls"}],
    [{"message": {"content": []}, "finish_reason": "stop"}],
    [{"message": {"content": "invalid JSON"}, "finish_reason": "stop"}],
])
def test_malformed_or_incomplete_response_returns_error(generator, hits, monkeypatch, choices):
    reply = response_for(choices=choices)
    monkeypatch.setattr(generator, "_request", lambda payload: reply)
    result = generator.generate("Question", hits)
    assert result["status"] == "error" and result["answer"] == ""
    assert result["usage"] == reply["usage"]


@pytest.mark.parametrize("response", [[], None, "not an object", {"error": {"message": "bad request"}}])
def test_invalid_response_envelope_returns_error(generator, hits, monkeypatch, response):
    monkeypatch.setattr(generator, "_request", lambda payload: response)
    assert generator.generate("Question", hits)["status"] == "error"


def test_legacy_limit_and_prompt_only_json(generator, hits, monkeypatch):
    generator.token_parameter = "max_tokens"
    generator.json_mode = False

    def request(payload):
        assert payload["max_tokens"] == 256 and "max_completion_tokens" not in payload
        assert "response_format" not in payload
        assert "JSON" in payload["messages"][0]["content"]
        return response_for(usage=None)

    monkeypatch.setattr(generator, "_request", request)
    result = generator.generate("Question", hits)
    assert result["status"] == "answered" and result["usage"] is None


def test_environment_key_and_explicit_override(monkeypatch):
    monkeypatch.setenv("CUSTOM_LLM_KEY", "environment-key")
    generator = APIGenerator("model", "https://provider.example/v1/", count_tokens=count_tokens,
                             api_key_env="CUSTOM_LLM_KEY")
    assert generator._api_key == "environment-key"
    assert generator.base_url == "https://provider.example/v1"
    explicit = APIGenerator("model", "https://provider.example/v1", count_tokens=count_tokens,
                            api_key="explicit-key", api_key_env="CUSTOM_LLM_KEY")
    assert explicit._api_key == "explicit-key"


@pytest.mark.parametrize("settings", [
    {"base_url": "http://provider.example/v1"}, {"base_url": "file:///tmp/api"},
    {"base_url": "https://user:password@provider.example/v1"},
    {"base_url": "https://provider.example/v1?key=secret"},
    {"base_url": "https://provider.example/v1#fragment"},
    {"model": " "}, {"count_tokens": None}, {"api_key": ""}, {"api_key": "secret\nkey"},
    {"num_ctx": 256}, {"num_ctx": True}, {"max_new_tokens": 0},
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")},
    {"token_parameter": "num_predict"}, {"json_mode": "true"},
])
def test_invalid_configuration_is_rejected(settings):
    options = {"model": "test-model", "base_url": "https://provider.example/v1",
               "count_tokens": count_tokens, "api_key": "test-key"}
    with pytest.raises(ValueError):
        APIGenerator(**{**options, **settings})


def test_missing_environment_key_is_rejected(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        APIGenerator("model", "https://provider.example/v1", count_tokens=count_tokens)


@pytest.mark.parametrize("base_url", ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"])
def test_local_api_endpoints_are_supported(base_url):
    generator = APIGenerator("model", base_url, count_tokens=count_tokens, api_key="test-key")
    assert generator.base_url == base_url


def test_http_request_encoding_headers_timeout_and_no_redirect(generator, hits, monkeypatch):
    class Reply(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://provider.example/v1/chat/completions"
            assert request.get_method() == "POST"
            assert request.get_header("Authorization") == "Bearer test-secret-key"
            assert request.get_header("Content-type") == "application/json"
            assert timeout == generator.timeout
            payload = json.loads(request.data.decode("utf-8"))
            assert "Cơ sở dữ liệu" in payload["messages"][1]["content"]
            return Reply(json.dumps(response_for()).encode("utf-8"))

    def opener(handler):
        assert isinstance(handler, api_generation._NoRedirect)
        assert handler.redirect_request(None, None, 302, "redirect", {}, "https://other.example") is None
        return Opener()

    monkeypatch.setattr(api_generation, "build_opener", opener)
    assert generator.generate("Cơ sở dữ liệu là gì?", hits)["status"] == "answered"


@pytest.mark.parametrize("failure, expected", [
    (HTTPError("https://provider.example", 401, "test-secret-key", {}, BytesIO(b"test-secret-key")), "HTTP 401"),
    (HTTPError("https://provider.example", 429, "rate limited", {}, BytesIO()), "HTTP 429"),
    (HTTPError("https://provider.example", 503, "unavailable", {}, BytesIO()), "HTTP 503"),
    (HTTPError("https://provider.example", 302, "redirect", {}, BytesIO()), "HTTP 302"),
    (URLError("test-secret-key"), "Không kết nối"), (TimeoutError("test-secret-key"), "Không kết nối"),
    (IncompleteRead(b"partial response"), "Không kết nối"),
])
def test_transport_failure_is_reported_without_retry_or_key(generator, hits, monkeypatch, failure, expected):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise failure

    monkeypatch.setattr(api_generation, "build_opener", lambda handler: Opener())
    result = generator.generate("Question", hits)
    assert result["status"] == "error" and expected in result["error"]
    assert result["called_model"] and len(calls) == 1
    assert "test-secret-key" not in json.dumps(result)


def test_api_and_local_share_instructions_and_context(hits):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, ensure_ascii=False)

        def encode(self, text, **kwargs):
            return list(text)

    local = build_context("Question", hits, Tokenizer())
    api = build_context("Question", hits, count_tokens=count_tokens)
    assert local == {key: value for key, value in api.items() if key != "messages"}
    local_result = OllamaGenerator(Tokenizer()).generate("Question", [])
    api_result = APIGenerator("model", "https://provider.example/v1", count_tokens=count_tokens,
                              api_key="test-key").generate("Question", [])
    assert local_result.keys() <= api_result.keys()


def test_openai_counter_and_generation_share_model_settings(hits, monkeypatch):
    calls = []

    def post(self, path, payload):
        calls.append((path, payload))
        assert payload["model"] == "gpt-6-luna"
        if path == "/responses/input_tokens":
            assert [message["role"] for message in payload["input"]] == ["system", "user"]
            return {"input_tokens": 80}
        assert path == "/chat/completions" and payload["reasoning_effort"] == "none"
        return response_for()

    monkeypatch.setattr(APIGenerator, "_post", post)
    generator = create_api_generator("gpt-6-luna", "https://api.openai.com/v1",
                                     api_key="test-secret-key", reasoning_effort="none")
    result = generator.generate("What is a database?", hits)
    assert result["status"] == "answered" and result["citation_valid"]
    assert result["prompt_tokens"] == 80 and result["usage"]["prompt_tokens"] == 31
    assert [path for path, _ in calls] == [
        "/responses/input_tokens", "/responses/input_tokens", "/chat/completions"]
