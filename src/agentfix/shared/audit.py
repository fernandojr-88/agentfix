"""Append-only JSONL audit log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def append_audit_entry(audit_log_path: str, **fields: Any) -> None:
    """Append a single JSON line to the audit log."""
    path = Path(audit_log_path)
    ensure_dir(path.parent)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
