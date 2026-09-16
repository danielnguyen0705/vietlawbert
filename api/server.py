import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool

from api.providers import ProviderFailure, make_provider
from api.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse

logger = logging.getLogger("uvicorn.error.vietlawbert")


def create_app(mode=None, provider_factory=None):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    mode = mode or os.getenv("CHAT_MODE", "mock")
    if mode not in {"mock", "rag"}:
        raise ValueError("CHAT_MODE must be mock or rag")
    factory = provider_factory or make_provider
    busy = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.provider = None
        try:
            app.state.provider = await run_in_threadpool(factory, mode)
        except Exception:
            logger.exception("provider_init_failed mode=%s", mode)
        try:
            yield
        finally:
            if app.state.provider is not None:
                await run_in_threadpool(app.state.provider.close)

    app = FastAPI(title="VietLawBERT Chat API", version="1.0.0", lifespan=lifespan)
    app.state.provider = None

    report_dir = os.getenv("CHAT_DEMO_REPORT_DIR")
    if report_dir:
        @app.get("/demo", include_in_schema=False)
        async def demo_report():
            report = Path(report_dir) / "index.html"
            if not report.is_file():
                raise HTTPException(404, "Chưa có báo cáo demo")
            return FileResponse(report, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        ready = app.state.provider is not None
        payload = HealthResponse(
            mode=mode, ready=ready, provider="ready" if ready else "unavailable",
            model="not_required" if mode == "mock" else ("loaded" if ready else "unavailable"),
            databases={name: "not_required" if mode == "mock" else "not_checked"
                       for name in ("qdrant", "elasticsearch", "neo4j", "redis")},
        )
        return JSONResponse(payload.model_dump(), status_code=200 if ready else 503)

    @app.post("/api/v1/chat", response_model=ChatResponse, responses={
        code: {"model": ErrorResponse, "description": description}
        for code, description in {500: "Unexpected error", 502: "Generation failed", 503: "Provider unavailable or busy"}.items()
    })
    async def chat(request: ChatRequest):
        started = time.perf_counter()
        status = 200

        def invoke():
            # The lock lives inside the worker so cancellation cannot release it
            # while the synchronous model is still running.
            if mode == "rag" and not busy.acquire(blocking=False):
                raise HTTPException(503, "Hệ thống đang bận. Vui lòng gửi lại sau.")
            try:
                return app.state.provider.ask(request.query, request.top_k)
            finally:
                if mode == "rag":
                    busy.release()

        try:
            if app.state.provider is None:
                raise HTTPException(503, "Bộ xử lý chưa sẵn sàng.")
            result = await run_in_threadpool(invoke)
            return ChatResponse(**result, mode=mode, latency_ms=(time.perf_counter() - started) * 1000)
        except HTTPException as exc:
            status = exc.status_code
            raise
        except ProviderFailure:
            status = 502
            raise HTTPException(502, "Không thể tạo câu trả lời. Vui lòng gửi lại sau.") from None
        except Exception:
            status = 500
            logger.exception("chat_failed mode=%s", mode)
            raise HTTPException(500, "Có lỗi xử lý. Vui lòng gửi lại sau.") from None
        finally:
            logger.info("chat status=%s mode=%s latency_ms=%.1f", status, mode, (time.perf_counter() - started) * 1000)

    return app


app = create_app()
