"""
统一日志模块

用法:
    from graphrag_agent.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("检索完成", extra={"query": query, "elapsed": 1.23})
"""
import os
import sys
import logging
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """结构化日志格式：JSON 行（方便 grep / 日志系统采集）"""

    def format(self, record):
        log_entry = {
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        # 附加 extra 字段（如 query, elapsed, retry_count 等）
        for key in ("query", "elapsed", "retry_count", "component", "error",
                     "top_k", "num_results", "round", "method"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class ReadableFormatter(logging.Formatter):
    """开发模式可读格式"""

    def format(self, record):
        ts = datetime.now().strftime("%H:%M:%S")
        level = record.levelname[0]  # I / W / E / D
        msg = record.getMessage()
        extras = []
        for key in ("query", "elapsed", "retry_count", "component", "error",
                     "top_k", "num_results", "round", "method"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        return f"{ts} {level} {record.name}: {msg}{extra_str}"


# 全局配置（只初始化一次）
_initialized = False


def setup_logging():
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_format = os.getenv("LOG_FORMAT", "readable")  # "json" | "readable"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger("graphrag_agent")
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # 避免重复 handler
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if log_format == "json":
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(ReadableFormatter())
        root.addHandler(handler)

    # 抑制第三方库的噪音
    for lib in ("httpx", "httpcore", "neo4j", "openai", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取 logger 实例，首次调用自动初始化"""
    setup_logging()
    return logging.getLogger(name)
