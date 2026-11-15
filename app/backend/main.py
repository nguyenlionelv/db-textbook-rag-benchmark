"""Serve a small HTTP API around the existing textbook RAG pipeline."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backend.config import Settings
from app.backend.runtime import Runtime
from app.backend.schemas import Answer, AskRequest, public_answer

logger = logging.getLogger(__name__)


def create_app(settings=None, runtime=None):
    """Create an application with one resource owner and one inference slot."""
    settings = settings or Settings.from_env()
    runtime = runtime or Runtime(settings)

    @asynccontextmanager
    async def lifespan(application):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        application.state.executor = executor
        application.state.gate = asyncio.Lock()
        application.state.requests = deque()
        startup = loop.run_in_executor(executor, runtime.load)
        try:
            yield
        finally:
            await asyncio.shield(startup)
            await loop.run_in_executor(executor, runtime.close)
            executor.shutdown(wait=True)

    application = FastAPI(title="DB Notes API", version="1.0.0", lifespan=lifespan)
    application.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.get("/api/health")
    async def health():
        return {"status": runtime.status, "ready": runtime.status == "ready",
                "busy": application.state.gate.locked()}

    @application.get("/api/models")
    async def models():
        options = [{**option, "available": option["available"] and runtime.status == "ready"}
                   for option in runtime.models]
        default = next((option["backend"] for option in options if option["available"]), None)
        return {"models": options, "default_backend": default}

    @application.post("/api/ask", response_model=Answer)
    async def ask(body: AskRequest):
        if runtime.status != "ready":
            raise HTTPException(503, "Backend chưa sẵn sàng. Bạn vẫn có thể xem các ví dụ lưu sẵn.")
        if body.backend not in runtime.generators:
            raise HTTPException(503, "Model này chưa sẵn sàng trên máy chủ.")
        gate = application.state.gate
        if gate.locked():
            raise HTTPException(429, "Đang xử lý một câu hỏi khác. Vui lòng thử lại sau.", headers={"Retry-After": "10"})
        accepted = application.state.requests
        now = monotonic()
        while accepted and now - accepted[0] >= 3600:
            accepted.popleft()
        if len(accepted) >= settings.requests_per_hour:
            retry = max(1, int(3600 - (now - accepted[0])))
            raise HTTPException(429, "Demo đã đạt giới hạn lượt hỏi trong giờ này. Bạn có thể xem ví dụ lưu sẵn.",
                                headers={"Retry-After": str(retry)})
        async with gate:
            accepted.append(now)
            request_id = uuid4().hex[:12]
            loop = asyncio.get_running_loop()
            work = loop.run_in_executor(application.state.executor,
                                        lambda: runtime.answer(body.question, language=body.language, backend=body.backend))
            try:
                result = await asyncio.shield(work)
            except asyncio.CancelledError:
                await asyncio.shield(work)
                raise
            except Exception as error:
                logger.error("Request %s failed (%s)", request_id, type(error).__name__)
                raise HTTPException(503, "Không xử lý được câu hỏi. Vui lòng thử lại sau.") from None
            logger.info("Request %s: %s, %.2fs", request_id, result["status"], result["total_seconds"])
            return public_answer(result, backend=body.backend, model=runtime.generators[body.backend].model,
                                 request_id=request_id)

    frontend = Path(__file__).resolve().parents[1] / "frontend/dist"
    if frontend.is_dir():
        application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return application


app = create_app()
