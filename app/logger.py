"""
AI Call Logger — Logs every LLM invocation to files for debugging.

Each line format:
    2026-04-29 15:12:30 {"type": "generate", "messages_count": 5, "prompt": "...", ...}
"""

import json
from datetime import datetime
from pathlib import Path


# Log file paths (project root / logs / *.log)
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_REQUEST_LOG = _LOG_DIR / "request.log"
_RESPONSE_LOG = _LOG_DIR / "response.log"
_RESPONSE_TYPES = {"generate", "summarize", "diary"}


def _ensure_log_dir():
    """Create logs directory if it doesn't exist."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_ai_call(call_type: str, **kwargs):
    """
    Log an AI call to the appropriate log file.

    Request types → logs/request.log
    Response types → logs/response.log

    Args:
        call_type: Type of call (e.g. "generate", "summarize", "diary")
        **kwargs: Additional data to log (prompt, messages, response, etc.)
    """
    _ensure_log_dir()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"type": call_type, **kwargs}
    line = f"{timestamp} {json.dumps(log_entry, ensure_ascii=False)}\n"

    # Route to the correct log file
    log_file = _RESPONSE_LOG if call_type in _RESPONSE_TYPES else _REQUEST_LOG

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)
