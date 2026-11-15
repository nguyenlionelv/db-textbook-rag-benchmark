"""Keep the saved-index pipeline alive on one worker thread."""

from contextlib import ExitStack
import logging
import os

from src.rag_pipeline import RAGPipeline, open_rag_pipeline

logger = logging.getLogger(__name__)


class Runtime:
    """Own retrieval resources and configured generators for one app process."""

    def __init__(self, settings):
        self.settings = settings
        self.status = "starting"
        self.generators = {}
        self.models = []
        self.retriever = None
        self.resources = ExitStack()

    def load(self):
        """Initialize once; leave examples usable if live setup is unavailable."""
        if self.settings.mode == "examples":
            self.status = "examples"
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            torch.set_num_threads(4)
            self._load_generators()
            if not self.generators:
                self.status = "unavailable"
                return
            pipeline = self.resources.enter_context(open_rag_pipeline(
                self.settings.index_manifest, next(iter(self.generators.values())),
                e5_cache=self.settings.e5_cache, reranker_cache=self.settings.reranker_cache,
            ))
            self.retriever = pipeline.retriever
            self.status = "ready"
        except Exception as error:
            logger.error("RAG initialization failed (%s)", type(error).__name__)
            self.resources.close()
            self.status = "unavailable"

    def _load_generators(self):
        """Enable only backends that have their required server configuration."""
        settings = self.settings
        configurations = [
            ("local", settings.local_model, settings.enable_local),
            ("api", settings.api_model, bool(settings.api_model or settings.api_url)),
        ]
        for backend, model, enabled in configurations:
            option = {"backend": backend, "model": model, "available": False,
                      "reason": "Chưa bật trong cấu hình máy chủ."}
            if enabled:
                try:
                    self.generators[backend] = self._generator(backend)
                    option.update(available=True, reason=None)
                except Exception as error:
                    logger.warning("%s backend unavailable (%s)", backend, type(error).__name__)
                    option["reason"] = ("Khởi động Ollama, kiểm tra model/tokenizer rồi khởi động lại backend."
                                        if backend == "local" else "Kiểm tra cấu hình LLM API trên máy chủ.")
            self.models.append(option)

    def _generator(self, backend):
        """Use the existing local and OpenAI-compatible generator contracts."""
        settings = self.settings
        common = {"num_ctx": settings.num_ctx, "max_new_tokens": settings.max_new_tokens,
                  "timeout": settings.timeout}
        if backend == "local":
            from transformers import AutoTokenizer
            from src.generation.textbook_generation import OllamaGenerator
            tokenizer = AutoTokenizer.from_pretrained(str(settings.local_tokenizer), local_files_only=True)
            generator = OllamaGenerator(tokenizer, model=settings.local_model,
                                        base_url=settings.ollama_url, **common)
            generator.check_model()
            return generator

        from src.generation.api_generation import create_api_generator

        return create_api_generator(
            settings.api_model, settings.api_url,
            token_counter=settings.api_token_counter, tokenizer_path=settings.api_tokenizer,
            api_key_env=settings.api_key_env, token_parameter=settings.api_token_parameter,
            reasoning_effort=settings.api_reasoning_effort, json_mode=settings.api_json_mode, **common,
        )

    def answer(self, question, *, language, backend):
        """Select a generator without mutating shared request state."""
        pipeline = RAGPipeline(self.retriever, self.generators[backend])
        return pipeline.answer(question, language=language)

    def close(self):
        """Close the Qdrant client on its owning worker thread."""
        self.resources.close()
        self.retriever = None
        self.generators.clear()
