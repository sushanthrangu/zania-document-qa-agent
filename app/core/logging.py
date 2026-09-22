import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format application logs as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
        ):
            value = getattr(record, field, None)

            if value is not None:
                log_record[field] = value

        if record.exc_info:
            log_record["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_record)


def configure_logging() -> None:
    """Configure application logging with JSON output."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when the application is imported
    # repeatedly during tests or development.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)