"""Generate source-linked answers through an OpenAI-compatible Chat API."""

import json
import math
import os
from http.client import HTTPException
from importlib import import_module
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.generation.textbook_generation import build_context, validate_answer


class _NoRedirect(HTTPRedirectHandler):
    """Keep authenticated requests on the configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class APIGenerator:
    """Generate RAG answers using POST /chat/completions without an SDK.

    Args:
        model: Model identifier supported by the provider.
        base_url: API root including its version prefix, such as /v1.
            HTTPS is required except for HTTP on localhost.
        count_tokens: Callable counting a message list with the provider's
            tokenizer and chat overhead. It must not use Qwen formatting for
            an unrelated model. Reported API usage is stored separately.
        api_key: Bearer token; otherwise read from api_key_env.
        api_key_env: Environment variable holding the token, default LLM_API_KEY.
        num_ctx: Model context capacity, including prompt and completion.
        max_new_tokens: Completion budget, including any reasoning tokens.
        timeout: HTTP timeout in seconds. Requests are not retried automatically.
        token_parameter: max_completion_tokens, or max_tokens for older APIs.
        json_mode: Request JSON object output; disable if unsupported.
        reasoning_effort: Optional provider reasoning setting, such as "none".

    generate() returns the same answer, source, status, and timing fields as
    OllamaGenerator, plus API messages, model identifiers, and token usage.
    Provider refusals and invalid/truncated outputs return status="error".
    Citation checks validate IDs, not whether the answer is supported by text.
    """

    def __init__(self, model, base_url, *, count_tokens, api_key=None,
                 api_key_env="LLM_API_KEY", num_ctx=4096, max_new_tokens=256,
                 timeout=180, token_parameter="max_completion_tokens", json_mode=True,
                 reasoning_effort=None):

        endpoint = urlparse(base_url)
        local_http = endpoint.scheme == "http" and endpoint.hostname in ("localhost", "127.0.0.1", "::1")
        if (not endpoint.hostname or (endpoint.scheme != "https" and not local_http)
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("base_url phải là HTTPS hoặc HTTP localhost, không chứa thông tin đăng nhập/query")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Cần tên model API")
        if not callable(count_tokens):
            raise ValueError("Cần count_tokens cho model API")
        if (type(num_ctx) is not int or type(max_new_tokens) is not int
                or not 0 < max_new_tokens < num_ctx):
            raise ValueError("Cần 0 < max_new_tokens < num_ctx")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout phải là số dương hữu hạn")
        if token_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("token_parameter phải là max_completion_tokens hoặc max_tokens")
        if type(json_mode) is not bool:
            raise ValueError("json_mode phải là bool")
        key = os.environ.get(api_key_env) if api_key is None else api_key
        if not isinstance(key, str) or not key.strip() or any(char.isspace() for char in key):
            raise ValueError("Cần API key hợp lệ qua api_key hoặc biến môi trường")

        self.model, self.base_url = model.strip(), base_url.rstrip("/")
        self.count_tokens, self.num_ctx = count_tokens, num_ctx
        self.max_new_tokens, self.timeout = max_new_tokens, timeout
        self.token_parameter, self.json_mode = token_parameter, json_mode
        self.reasoning_effort = reasoning_effort
        self._api_key = key

    def _request(self, payload):
        """Send one authenticated request without redirects or automatic retries."""
        return self._post("/chat/completions", payload)

    def _post(self, path, payload):
        """Post JSON to the configured API and decode its response."""
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            # Do not copy response bodies that may echo credentials or prompts.
            error.close()
            raise RuntimeError(f"LLM API HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError, HTTPException):
            raise RuntimeError("Không kết nối được LLM API hoặc đã hết thời gian chờ") from None

    def generate(self, query, hits, *, language="vi"):
        """Prepare context, request one answer, and validate its citations.

        query, hits, and language follow build_context(). Invalid inputs raise
        ValueError before a request. Empty context skips the API. The returned
        prompt_tokens is the caller's count; usage contains provider counts.
        """
        start = perf_counter()
        prepared = build_context(query, hits, num_ctx=self.num_ctx,
                                 max_new_tokens=self.max_new_tokens, language=language,
                                 count_tokens=self.count_tokens)
        result = {**prepared, "status": "insufficient_evidence", "answer":
                  "Chưa đủ bằng chứng trong context." if language == "vi" else "Insufficient evidence in context.",
                  "citations": [], "citation_valid": True, "called_model": False,
                  "raw_output": "", "model": self.model, "response_model": None,
                  "usage": None, "finish_reason": None, "response_id": None}
        if not prepared["sources"]:
            return {**result, "seconds": perf_counter() - start}

        payload = {"model": self.model, "messages": prepared["messages"], "stream": False,
                   self.token_parameter: self.max_new_tokens}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        result["called_model"] = True

        try:
            response = self._request(payload)
            result.update(usage=response.get("usage"), response_model=response.get("model"),
                          response_id=response.get("id"))
            choices = response["choices"]
            if len(choices) != 1:
                raise ValueError("Phản hồi API phải có đúng một choice")
            choice = choices[0]
            message = choice["message"]
            result["raw_output"] = message.get("content")
            result["finish_reason"] = choice.get("finish_reason")
            if result["finish_reason"] != "stop" or message.get("refusal") or message.get("tool_calls"):
                raise ValueError("API từ chối, chưa hoàn tất hoặc không trả đáp án văn bản")
            result.update(validate_answer(result["raw_output"], prepared["sources"]))
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as error:
            result.update(status="error", answer="", citations=[], citation_valid=False,
                          error=str(error).replace(self._api_key, "[REDACTED]"))
        return {**result, "seconds": perf_counter() - start}


def create_api_generator(model, base_url, *, tokenizer_path=None, token_counter=None, **options):
    """Use a configured counter, a local tokenizer, or OpenAI's token endpoint.

    OpenAI counts use the Responses input format. Chat Completions usage is
    retained separately because request-format overhead may differ.
    """
    if tokenizer_path and token_counter:
        raise ValueError("Choose either an API tokenizer or a token counter")

    if token_counter:
        module, name = token_counter.split(":", 1)
        count_tokens = getattr(import_module(module), name)
    elif tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

        def count_tokens(messages):
            return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    elif urlparse(base_url).hostname == "api.openai.com":
        # Count input with OpenAI instead of guessing the model's tokenizer.
        def count_tokens(messages):
            response = generator._post("/responses/input_tokens", {"model": model, "input": messages})
            return response["input_tokens"]
    else:
        raise ValueError("API requires --token-counter or a matching --tokenizer-path")

    generator = APIGenerator(model, base_url, count_tokens=count_tokens, **options)
    return generator
