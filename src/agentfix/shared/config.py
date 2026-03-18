"""Shared configuration models used by all agents."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class RuntimeSettings(BaseModel):
    dry_run: bool = False
    artifacts_dir: str = ""
    close_issue_after_push: bool = True
    push_after_commit: bool = True
    command_timeout_seconds: int = 1800
    require_clean_worktree: bool = True
    preserve_worktree: bool = True
    history_file: str = ""
    audit_log: str = ""


class GitHubSettings(BaseModel):
    enabled: bool = True
    repo_owner: str = ""
    repo_name: str = ""
    labels: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)


class JuniorLLMSettings(BaseModel):
    """Settings for the junior-tier LLM CLI (e.g. Codex CLI, aider, etc.)."""

    provider: str = "codex_cli"
    binary: str = "codex"
    model: str = ""
    approval_mode: str = "full-auto"
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    task_timeout_seconds: int = 1800


class SeniorLLMSettings(BaseModel):
    """Settings for the senior-tier LLM CLI (e.g. Claude Code CLI)."""

    enabled: bool = True
    binary: str = "claude"
    max_attempts: int = 2
    task_timeout_seconds: int = 2400
    base_context_lines: int = 4000
    expanded_context_lines: int = 8000
    env: dict[str, str] = Field(default_factory=dict)


class EscalationSettings(BaseModel):
    enabled: bool = True
    max_junior_attempts: int = 2
    max_senior_attempts: int = 2
    escalation_label: str = "needs-human-review"
    agent_fixable_label: str = "agent-fixable"


class QualityGateSettings(BaseModel):
    run_regression: bool = True
    regression_script: str = "tests/run_all_tests.sh"
    regression_timeout_seconds: int = 600
    run_static_analysis: bool = False
    static_analysis_command: str = ""
    static_analysis_args: list[str] = Field(default_factory=list)


def expand_env_values(value: Any) -> Any:
    """Recursively expand ${VAR} references in config values."""
    if isinstance(value, dict):
        return {k: expand_env_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_values(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
