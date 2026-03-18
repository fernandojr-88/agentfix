"""Persistent history — tracks per-item outcomes across runs to prevent infinite loops."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import ensure_dir


class HistoryEntry:
    __slots__ = ("fingerprint", "run_id", "outcome", "timestamp", "failure_reason", "tier", "extra")

    def __init__(
        self,
        fingerprint: str,
        run_id: str,
        outcome: str,
        timestamp: str = "",
        failure_reason: str = "",
        tier: str = "",
        **extra: Any,
    ) -> None:
        self.fingerprint = fingerprint
        self.run_id = run_id
        self.outcome = outcome
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.failure_reason = failure_reason
        self.tier = tier
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "run_id": self.run_id,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "failure_reason": self.failure_reason,
        }
        if self.tier:
            d["tier"] = self.tier
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        known = {"fingerprint", "run_id", "outcome", "timestamp", "failure_reason", "tier"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            fingerprint=str(data.get("fingerprint", "")),
            run_id=str(data.get("run_id", "")),
            outcome=str(data.get("outcome", "")),
            timestamp=str(data.get("timestamp", "")),
            failure_reason=str(data.get("failure_reason", "")),
            tier=str(data.get("tier", "")),
            **extra,
        )


class ItemHistory:
    """Persistent history of fix/generation attempts, backed by a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: list[HistoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [HistoryEntry.from_dict(item) for item in raw]
        except (json.JSONDecodeError, KeyError, TypeError):
            self._entries = []

    def _save(self) -> None:
        ensure_dir(self._path.parent)
        self._path.write_text(
            json.dumps([e.to_dict() for e in self._entries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def record(self, fingerprint: str, run_id: str, outcome: str, **kwargs: Any) -> None:
        """Append a new history entry and persist."""
        entry = HistoryEntry(fingerprint=fingerprint, run_id=run_id, outcome=outcome, **kwargs)
        self._entries.append(entry)
        self._save()

    def attempts_for(self, fingerprint: str) -> list[HistoryEntry]:
        return [e for e in self._entries if e.fingerprint == fingerprint]

    def failure_count(self, fingerprint: str) -> int:
        return sum(1 for e in self._entries if e.fingerprint == fingerprint and e.outcome == "failed")

    def junior_failure_count(self, fingerprint: str) -> int:
        return sum(
            1 for e in self._entries
            if e.fingerprint == fingerprint and e.outcome == "failed" and e.tier == "junior"
        )

    def senior_failure_count(self, fingerprint: str) -> int:
        return sum(
            1 for e in self._entries
            if e.fingerprint == fingerprint and e.outcome == "failed" and e.tier == "senior"
        )

    def should_skip(self, fingerprint: str, max_attempts: int = 2) -> bool:
        return self.failure_count(fingerprint) >= max_attempts

    def was_completed(self, fingerprint: str) -> bool:
        return any(
            e.fingerprint == fingerprint and e.outcome in ("fixed", "completed")
            for e in self._entries
        )

    def was_escalated(self, fingerprint: str) -> bool:
        return any(e.fingerprint == fingerprint and e.outcome == "escalated" for e in self._entries)

    def last_failure_details(self, fingerprint: str) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries if e.fingerprint == fingerprint and e.outcome == "failed"]

    @property
    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)
