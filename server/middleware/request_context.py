"""
请求追踪中间件

为每个 HTTP 请求生成唯一 request_id，记录 method、path、status、duration。
配合结构化日志，可以在日志中追踪完整的请求链路。
"""
import time
import uuid
import asyncio
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """为每个请求生成 request_id、记录耗时和状态码"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.time()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            duration = round(time.time() - start, 3)
            # 把 request_id 回传给客户端，方便排查
            response.headers["X-Request-ID"] = request_id
            logger.info(
                f"{method} {path} -> {response.status_code}",
                extra={
                    "elapsed": duration,
                    "method": method,
                    "component": "http",
                },
            )
            return response

        except Exception as e:
            duration = round(time.time() - start, 3)
            logger.error(
                f"{method} {path} 未捕获异常",
                extra={
                    "error": str(e),
                    "elapsed": duration,
                    "method": method,
                    "component": "http",
                },
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "request_id": request_id,
                    "detail": str(e),
                },
                headers={"X-Request-ID": request_id},
            )


class TimeoutMiddleware(BaseHTTPMiddleware):
    """全局请求超时，防止慢请求长期占用 worker"""

    def __init__(self, app, timeout_seconds: int = 120, skip_paths: tuple = ()):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        self.skip_paths = skip_paths  # SSE 流式端点跳过

    async def dispatch(self, request: Request, call_next):
        # 流式端点不设超时（SSE 可能跑很久）
        if any(request.url.path.startswith(p) for p in self.skip_paths):
            return await call_next(request)

        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            request_id = getattr(request.state, "request_id", "-")
            logger.warning(
                f"请求超时 ({self.timeout_seconds}s) {request.method} {request.url.path}",
                extra={"component": "http", "method": request.method},
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Gateway Timeout",
                    "request_id": request_id,
                    "detail": f"请求处理超过 {self.timeout_seconds}s，已终止",
                },
            )
