"""Shared utility functions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def dump_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str, limit: int = 60) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    lowered = lowered[:limit].strip("-")
    return lowered or "item"


def excerpt(text: str, max_chars: int = 1500) -> str:
    compact = text.strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n...[truncated]"


def safe_read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")
