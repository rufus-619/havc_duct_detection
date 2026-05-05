import logging
import json
import datetime
import traceback
from contextvars import ContextVar
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="unknown")

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "trace_id": trace_id_var.get(),
            "event": record.getMessage()
        }
        
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
            
        if record.exc_info:
            log_obj["exception"] = "".join(traceback.format_exception(*record.exc_info))
            
        return json.dumps(log_obj)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        from app.core.config import settings
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger

class StructuredLogger:
    def __init__(self, name: str):
        self._logger = get_logger(name)
        
    def info(self, event: str, **kwargs):
        self._logger.info(event, extra={"extra_data": kwargs})
        
    def error(self, event: str, **kwargs):
        self._logger.error(event, extra={"extra_data": kwargs})
        
    def warning(self, event: str, **kwargs):
        self._logger.warning(event, extra={"extra_data": kwargs})
        
    def debug(self, event: str, **kwargs):
        self._logger.debug(event, extra={"extra_data": kwargs})
        
logger = StructuredLogger("hvac_analyzer")
