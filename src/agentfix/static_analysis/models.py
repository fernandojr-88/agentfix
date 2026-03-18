"""Models specific to the Static Analysis Fixer agent."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class WarningCandidate(BaseModel):
    """A single static analysis warning that may be auto-fixed."""

    id: str
    severity: str
    file: str
    line: int
    column: int | None = None
    message: str
    verbose: str = ""
    cwe: str = ""
    symbol: str = ""
    fingerprint: str = ""

    @model_validator(mode="after")
    def finalize(self) -> "WarningCandidate":
        if not self.fingerprint:
            self.fingerprint = f"{self.id}:{self.file}:{self.line}"
        return self
