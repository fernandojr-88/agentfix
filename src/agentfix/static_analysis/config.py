"""Configuration for the Static Analysis Fixer agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..shared.config import (
    EscalationSettings,
    GitHubSettings,
    JuniorLLMSettings,
    RuntimeSettings,
    expand_env_values,
)


class StaticAnalysisRuntimeSettings(RuntimeSettings):
    max_fix_attempts: int = 1
    max_retry_per_warning: int = 2
    fail_on_missing_tests: bool = True


class StaticAnalysisToolSettings(BaseModel):
    """Settings for the static analysis tool (e.g. cppcheck, clang-tidy, pylint, ESLint, etc.)."""

    command: str = ""
    output_format: str = "xml"  # "xml" | "json" | "text"
    source_paths: list[str] = Field(default_factory=lambda: ["src"])
    extra_args: list[str] = Field(default_factory=list)
    allowed_severities: list[str] = Field(
        default_factory=lambda: ["error", "warning", "performance", "portability", "style"]
    )
    ignore_warning_ids: list[str] = Field(default_factory=list)


class BuildSettings(BaseModel):
    setup_command: str = ""
    command: str = ""
    env: dict[str, str] = Field(default_factory=dict)


class TestSettings(BaseModel):
    commands: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SelectionPolicy(BaseModel):
    prefer_severities: list[str] = Field(
        default_factory=lambda: ["error", "warning", "performance", "portability", "style"]
    )
    ignore_paths: list[str] = Field(default_factory=list)
    ignore_warning_ids: list[str] = Field(default_factory=list)


class StaticAnalysisEscalationSettings(EscalationSettings):
    rules: list[dict[str, str]] = Field(default_factory=list)


class AgentConfig(BaseModel):
    project_name: str
    repo_path: str = "."
    branch_prefix: str = "fix/static-analysis"
    git_remote: str = "origin"
    runtime: StaticAnalysisRuntimeSettings = Field(default_factory=StaticAnalysisRuntimeSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    junior_llm: JuniorLLMSettings = Field(default_factory=JuniorLLMSettings)
    static_analysis: StaticAnalysisToolSettings = Field(default_factory=StaticAnalysisToolSettings)
    build: BuildSettings = Field(default_factory=BuildSettings)
    tests: TestSettings = Field(default_factory=TestSettings)
    selection_policy: SelectionPolicy = Field(default_factory=SelectionPolicy)
    escalation: StaticAnalysisEscalationSettings = Field(default_factory=StaticAnalysisEscalationSettings)


def load_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = expand_env_values(raw)
    config = AgentConfig.model_validate(raw)
    if not config.github.repo_name:
        config.github.repo_name = config.project_name
    return config
