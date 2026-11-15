"""Answer questions using a saved textbook index and a configured generator."""

from contextlib import closing, contextmanager
import json
from pathlib import Path
from time import perf_counter

RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class RAGPipeline:
    """Reuse one retriever and either a local or API generator across questions."""

    def __init__(self, retriever, generator, *, top_k=5, candidate_k=20):
        if not 0 < top_k <= candidate_k:
            raise ValueError("Require 0 < top_k <= candidate_k")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.candidate_k = candidate_k

    def answer(self, query, *, language="vi"):
        """Return the answer, retrieved hits, selected sources, and stage timings.

        Handled failures have status='error' and retain their failing stage.
        Citation validity checks source IDs; it does not establish factuality.
        """
        start = perf_counter()
        hits, retrieval_seconds, generation_seconds = [], 0.0, 0.0
        stage = "retrieval"
        result = {"status": "error", "answer": "", "citations": [], "sources": [],
                  "citation_valid": False, "called_model": False}
        try:
            hits = self.retriever.retrieve(query, top_k=self.top_k, candidate_k=self.candidate_k)
            retrieval_seconds = perf_counter() - start
            stage = "generation"
            result = self.generator.generate(query, hits, language=language)
        except (ValueError, RuntimeError, OSError, TypeError) as error:
            result.update(status="error", error=str(error))
        elapsed = perf_counter() - start
        if stage == "retrieval":
            retrieval_seconds = elapsed
        else:
            generation_seconds = elapsed - retrieval_seconds
        return {**result, "query": query, "language": language, "retrieved_hits": hits,
                "retrieval_seconds": retrieval_seconds, "generation_seconds": generation_seconds,
                "total_seconds": elapsed, "error_stage": stage if result["status"] == "error" else None}


@contextmanager
def open_rag_pipeline(index_manifest, generator, *, top_k=5, candidate_k=20,
                      e5_cache=None, reranker_cache=None, reranker_max_length=512):
    """Load cached retrieval models and open an existing Qdrant index once.

    The caller closes the context after serving all questions. No document
    parsing, document embedding, collection creation, or upsert is performed.
    """
    from qdrant_client import QdrantClient
    from src.ingestion.textbook_embedder import E5Embedder
    from src.retrieval.textbook_retrieval import CrossEncoderReranker, TextbookRetriever

    profile = json.loads(Path(index_manifest).read_text(encoding="utf-8"))
    assert profile["status"] == "complete", "The saved index is incomplete"
    assert Path(profile["qdrant_path"]).is_dir(), "The saved Qdrant directory is missing"
    embedder = E5Embedder(model_name=profile["model"], revision=profile["model_revision"],
                          cache_folder=str(e5_cache) if e5_cache else None, local_files_only=True)
    reranker = CrossEncoderReranker(revision=RERANKER_REVISION, max_length=reranker_max_length,
                                    cache_folder=str(reranker_cache) if reranker_cache else None)
    with closing(QdrantClient(path=profile["qdrant_path"])) as client:
        retriever = TextbookRetriever(client, profile["collection"], embedder, reranker,
                                      embedding_profile=profile)
        assert len(retriever.chunks) == profile["chunks"], "Saved index count mismatch"
        yield RAGPipeline(retriever, generator, top_k=top_k, candidate_k=candidate_k)
