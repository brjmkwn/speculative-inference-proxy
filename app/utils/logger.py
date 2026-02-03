import sys
import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_obj.update(record.extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str = "proxy", level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        log.addHandler(handler)
        log.propagate = False
        
    return log


logger = get_logger()
