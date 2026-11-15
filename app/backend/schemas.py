"""Public request and response contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

Backend = Literal["local", "api"]


class AskRequest(BaseModel):
    """One independent textbook question."""

    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    language: Literal["vi", "en"] = "vi"
    backend: Backend = "local"


class Source(BaseModel):
    """A selected context chunk with PDF pages numbered from one."""

    source_id: str
    point_id: str
    text: str
    pdf_pages: list[int]
    chapter: str
    section: str
    title: str


class Answer(BaseModel):
    """Fields needed by the UI; excludes private prompts and paths."""

    request_id: str
    query: str
    language: Literal["vi", "en"]
    backend: Backend
    model: str
    status: Literal["answered", "insufficient_evidence", "error"]
    answer: str
    citations: list[str]
    sources: list[Source]
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    error: str | None = None


def public_answer(result, *, backend, model, request_id):
    """Resolve source text from retrieved hits without exporting raw metadata."""
    hits = {str(hit["point_id"]): hit for hit in result["retrieved_hits"]}
    sources = []
    for source in result["sources"]:
        metadata = source["metadata"]
        sources.append(Source(
            source_id=source["source_id"], point_id=str(source["point_id"]),
            text=hits[str(source["point_id"])]["page_content"],
            pdf_pages=[page + 1 for page in source["pdf_pages"]],
            chapter=str(metadata.get("chapter_number") or ""),
            section=str(metadata.get("section_number") or ""),
            title=metadata.get("section_title") or metadata.get("chapter_title") or "Textbook",
        ))
    return Answer(
        request_id=request_id, query=result["query"], language=result["language"], backend=backend,
        model=model, status=result["status"], answer=result["answer"], sources=sources,
        citations=[source["source_id"] for source in result["citations"]],
        retrieval_seconds=result["retrieval_seconds"], generation_seconds=result["generation_seconds"],
        total_seconds=result["total_seconds"],
        error="Không xử lý được câu hỏi. Hãy thử câu ngắn hơn hoặc kiểm tra dịch vụ model."
              if result["status"] == "error" else None,
    )
