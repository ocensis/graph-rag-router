import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from routers import api_router
from server_config.database import get_db_manager
from server_config.settings import UVICORN_CONFIG
from services.agent_service import agent_manager
from middleware.request_context import RequestTracingMiddleware, TimeoutMiddleware

from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(
    title="知识图谱问答系统",
    description="基于知识图谱的智能问答系统后端API",
)

# 中间件顺序：先 Timeout 包住请求，再 Tracing 记录（顺序相反：先注册的后执行）
app.add_middleware(
    TimeoutMiddleware,
    timeout_seconds=120,
    skip_paths=("/chat/stream",),  # 流式端点不受超时限制
)
app.add_middleware(RequestTracingMiddleware)

# 路由
app.include_router(api_router)


# ==================== 全局异常处理器 ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic 请求体校验失败 → 422"""
    request_id = getattr(request.state, "request_id", "-")
    logger.warning(
        f"请求参数校验失败 {request.url.path}",
        extra={"error": str(exc.errors())[:200], "component": "http"},
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "request_id": request_id,
            "detail": exc.errors(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTPException 统一格式"""
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "request_id": request_id,
            "detail": exc.detail,
        },
    )


# ==================== 生命周期 ====================

# 获取数据库连接
db_manager = get_db_manager()
driver = db_manager.driver


@app.on_event("startup")
async def startup_event():
    logger.info("服务启动", extra={"component": "server"})


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("服务关闭中", extra={"component": "server"})
    agent_manager.close_all()
    if driver:
        driver.close()
        logger.info("Neo4j 连接已关闭", extra={"component": "server"})


# 启动
if __name__ == "__main__":
    uvicorn.run("main:app", **UVICORN_CONFIG)
