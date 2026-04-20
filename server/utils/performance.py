import time
import functools

from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)


def measure_performance(endpoint_name):
    """
    API 性能测量装饰器（结构化日志）

    Args:
        endpoint_name: API端点名称
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = round(time.time() - start_time, 4)
                logger.info(
                    f"API 性能 - {endpoint_name}",
                    extra={"elapsed": duration, "component": f"api.{endpoint_name}"},
                )
                return result
            except Exception as e:
                duration = round(time.time() - start_time, 4)
                logger.error(
                    f"API 异常 - {endpoint_name}",
                    extra={
                        "elapsed": duration,
                        "error": str(e),
                        "component": f"api.{endpoint_name}",
                    },
                )
                raise
        return wrapper
    return decorator
