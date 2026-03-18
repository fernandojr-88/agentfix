"""Configuration for the Test Generator agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..shared.config import (
    EscalationSettings,
    GitHubSettings,
    JuniorLLMSettings,
    QualityGateSettings,
    RuntimeSettings,
    SeniorLLMSettings,
    expand_env_values,
)


class TestGenRuntimeSettings(RuntimeSettings):
    max_fix_attempts: int = 3
    max_retry_per_module: int = 3


class CoverageSettings(BaseModel):
    """Settings for code coverage measurement."""

    compiler: str = "gcc"
    compiler_flags: list[str] = Field(default_factory=lambda: ["-std=c99", "-Wall", "-Wextra"])
    coverage_flags: list[str] = Field(default_factory=lambda: ["--coverage"])
    source_paths: list[str] = Field(default_factory=lambda: ["src"])
    test_framework_path: str = "tests/vendor/unity"
    test_output_dir: str = "tests/unit"
    manifest_path: str = ""
    include_paths: list[str] = Field(default_factory=list)


class SelectionPolicy(BaseModel):
    prefer_tiers: list[str] = Field(
        default_factory=lambda: ["pure_logic", "mock_able", "hw_dependent"]
    )
    ignore_paths: list[str] = Field(default_factory=list)
    min_coverage_increase: float = 1.0


class TestGenEscalationSettings(EscalationSettings):
    senior_review_label: str = "senior-review"
    max_attempts_before_escalate: int = 3


class TestGenQualityGateSettings(QualityGateSettings):
    require_negative_tests: bool = True


class AgentConfig(BaseModel):
    project_name: str
    repo_path: str = "."
    branch_prefix: str = "test/unit"
    git_remote: str = "origin"
    runtime: TestGenRuntimeSettings = Field(default_factory=TestGenRuntimeSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    junior_llm: JuniorLLMSettings = Field(default_factory=JuniorLLMSettings)
    senior_llm: SeniorLLMSettings = Field(default_factory=SeniorLLMSettings)
    coverage: CoverageSettings = Field(default_factory=CoverageSettings)
    selection_policy: SelectionPolicy = Field(default_factory=SelectionPolicy)
    escalation: TestGenEscalationSettings = Field(default_factory=TestGenEscalationSettings)
    quality_gates: TestGenQualityGateSettings = Field(default_factory=TestGenQualityGateSettings)


def load_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = expand_env_values(raw)
    config = AgentConfig.model_validate(raw)
    if not config.github.repo_name:
        config.github.repo_name = config.project_name
    return config
