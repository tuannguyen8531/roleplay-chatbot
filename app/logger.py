"""
AI Call Logger — Logs every LLM invocation to files for debugging.

Each line format:
    2026-04-29 15:12:30 {"direction": "request", "type": "generate", "call_id": "...", ...}
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


# Log file paths (project root / logs / *.log)
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_REQUEST_LOG = _LOG_DIR / "request.log"
_RESPONSE_LOG = _LOG_DIR / "response.log"


def _ensure_log_dir():
    """Create logs directory if it doesn't exist."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def new_call_id() -> str:
    """Create a correlation id for one request/response pair."""
    return uuid.uuid4().hex


def _write_ai_log(log_file: Path, direction: str, call_type: str, call_id: str, **kwargs):
    _ensure_log_dir()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "direction": direction,
        "type": call_type,
        "call_id": call_id,
        **kwargs,
    }
    line = f"{timestamp} {json.dumps(log_entry, ensure_ascii=False)}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def log_ai_request(call_type: str, call_id: str | None = None, **kwargs) -> str:
    """
    Log an AI request to logs/request.log and return its correlation id.

    Args:
        call_type: Type of call (e.g. "generate", "summarize", "diary")
        call_id: Optional existing id for the request/response pair
        **kwargs: Additional request data to log (prompt, messages, etc.)
    """
    call_id = call_id or new_call_id()
    _write_ai_log(_REQUEST_LOG, "request", call_type, call_id, **kwargs)
    return call_id


def log_ai_response(call_type: str, call_id: str, **kwargs):
    """
    Log an AI response to logs/response.log.

    Args:
        call_type: Type of call (e.g. "generate", "summarize", "diary")
        call_id: Correlation id returned by log_ai_request
        **kwargs: Additional response data to log (response, summary, facts, etc.)
    """
    _write_ai_log(_RESPONSE_LOG, "response", call_type, call_id, **kwargs)


def log_ai_call(call_type: str, **kwargs):
    """
    Log a non-LLM-pair event to logs/request.log.

    Kept for auxiliary events such as RAG search metadata that do not have a
    matching model response.
    """
    call_id = kwargs.pop("call_id", None) or new_call_id()
    _write_ai_log(_REQUEST_LOG, "event", call_type, call_id, **kwargs)
    return call_id
