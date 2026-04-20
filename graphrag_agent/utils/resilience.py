"""
弹性工具：retry with exponential backoff + timeout

用法:
    from graphrag_agent.utils.resilience import retry_async, retry_sync, async_timeout

    # 同步 retry
    @retry_sync(max_retries=3, base_delay=1.0)
    def call_llm(prompt):
        return llm.invoke(prompt)

    # 异步 retry
    @retry_async(max_retries=3, base_delay=1.0)
    async def call_llm_async(prompt):
        return await llm.ainvoke(prompt)

    # 异步超时
    result = await async_timeout(coro, timeout_seconds=30)
"""
import time
import asyncio
import functools
import random
from typing import TypeVar, Callable, Any

from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    同步 retry 装饰器，exponential backoff + jitter

    参数:
        max_retries: 最大重试次数
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        retryable_exceptions: 可重试的异常类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} 失败，已重试 {max_retries} 次",
                            extra={"error": str(e), "retry_count": attempt,
                                   "method": func.__name__},
                        )
                        raise
                    # exponential backoff + jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(
                        f"{func.__name__} 第 {attempt+1} 次失败，{delay:.1f}s 后重试",
                        extra={"error": str(e), "retry_count": attempt + 1,
                               "method": func.__name__},
                    )
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    异步 retry 装饰器，exponential backoff + jitter
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} 失败，已重试 {max_retries} 次",
                            extra={"error": str(e), "retry_count": attempt,
                                   "method": func.__name__},
                        )
                        raise
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(
                        f"{func.__name__} 第 {attempt+1} 次失败，{delay:.1f}s 后重试",
                        extra={"error": str(e), "retry_count": attempt + 1,
                               "method": func.__name__},
                    )
                    await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


async def async_timeout(coro, timeout_seconds: float = 30.0, default=None):
    """
    给任意 coroutine 加超时，超时返回 default 而非抛异常

    用法:
        result = await async_timeout(llm.ainvoke(prompt), timeout_seconds=30)
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"操作超时 ({timeout_seconds}s)")
        return default
