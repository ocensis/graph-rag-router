"""
Langfuse 可观测性集成

用法:
    from graphrag_agent.utils.langfuse_client import get_langfuse_handler

    handler = get_langfuse_handler()
    if handler:
        chain.invoke(inputs, config={"callbacks": [handler]})

未配置 LANGFUSE_* 环境变量时，get_langfuse_handler() 返回 None，
调用方需要判空，避免影响正常流程。
"""
import os
from typing import Optional

from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)

_handler = None
_initialized = False


def get_langfuse_handler(trace_name: Optional[str] = None,
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         tags: Optional[list] = None):
    """
    返回 Langfuse CallbackHandler。

    关键: 如果当前在 @observe 装饰的函数栈内，返回 context 绑定的 handler
         → 下游调用会自动嵌套到父 trace 下（不再是独立 trace）
         否则创建/返回全局 handler（独立 trace 行为）
    """
    # 优先检查 langfuse_context（Router.ask 包了 @observe 时这里会拿到绑定 handler）
    # 注意: get_current_langchain_handler() 在没有 @observe 栈时会打 "No observation found" warning
    #       我们大部分调用不在 @observe 栈内（用 CallbackHandler 模式），所以把 langfuse decorator
    #       logger 静音到 ERROR 级别，避免刷屏
    try:
        import logging as _logging
        _lf_logger = _logging.getLogger("langfuse")
        if _lf_logger.level < _logging.ERROR:
            _lf_logger.setLevel(_logging.ERROR)

        from langfuse.decorators import langfuse_context
        ctx_handler = langfuse_context.get_current_langchain_handler()
        if ctx_handler is not None:
            return ctx_handler
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"langfuse_context 检查失败: {e}")

    # 没有活跃 context → 返回全局 handler（独立 trace）
    global _handler, _initialized

    if not _initialized:
        _initialized = True
        try:
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            host = os.getenv("LANGFUSE_BASE_URL")

            if not (secret_key and public_key and host):
                logger.info("Langfuse 未配置，跳过 trace", extra={"component": "langfuse"})
                _handler = None
                return None

            from langfuse.callback import CallbackHandler
            _handler = CallbackHandler(
                secret_key=secret_key,
                public_key=public_key,
                host=host,
            )
            logger.info(f"Langfuse 初始化完成: {host}", extra={"component": "langfuse"})
        except Exception as e:
            logger.warning(f"Langfuse 初始化失败，跳过 trace: {e}",
                           extra={"component": "langfuse", "error": str(e)})
            _handler = None

    return _handler


_langfuse_client = None


def get_langfuse_client():
    """
    返回 Langfuse SDK client 实例（不是 CallbackHandler）。
    用于直接调 langfuse.trace(...).update() 这类 API，更新 trace 的 tag/metadata。
    """
    global _langfuse_client
    if _langfuse_client is None:
        try:
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            host = os.getenv("LANGFUSE_BASE_URL")
            if not (secret_key and public_key and host):
                return None
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                secret_key=secret_key, public_key=public_key, host=host,
                timeout=15,
            )
        except Exception as e:
            logger.debug(f"Langfuse client 初始化失败: {e}")
            return None
    return _langfuse_client


def build_callback_config(trace_name: str,
                          session_id: Optional[str] = None,
                          tags: Optional[list] = None,
                          metadata: Optional[dict] = None) -> dict:
    """
    构建带 Langfuse callback 的 config 字典

    用法:
        config = build_callback_config("agentic_search", session_id="demo")
        chain.invoke(inputs, config=config)
    """
    handler = get_langfuse_handler()
    if handler is None:
        return {}

    run_metadata = {
        "langfuse_session_id": session_id,
        "langfuse_tags": tags or [],
    }
    if metadata:
        run_metadata.update(metadata)

    return {
        "callbacks": [handler],
        "run_name": trace_name,
        "metadata": run_metadata,
    }


def flush_langfuse():
    """强制 flush 所有待上报的 trace（脚本退出前调用）"""
    global _handler
    if _handler is not None:
        try:
            _handler.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush 失败: {e}",
                           extra={"component": "langfuse", "error": str(e)})
