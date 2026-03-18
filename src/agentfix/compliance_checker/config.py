"""Configuration for the Compliance Checker agent."""

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


class ComplianceRuntimeSettings(RuntimeSettings):
    task_plan_file: str = ""


class ComplianceSettings(BaseModel):
    """Settings specific to compliance/regulatory requirements."""

    docs_root: str = "docs/requirements"
    gap_analysis: str = "gap_analysis.md"
    readiness_doc: str = "readiness.md"
    skip_requirements: list[str] = Field(default_factory=list)
    partial_requirements: list[str] = Field(default_factory=list)
    critical_modules: list[str] = Field(default_factory=list)


class ComplianceQualityGateSettings(QualityGateSettings):
    run_power10: bool = False


class ComplianceEscalationSettings(EscalationSettings):
    critical_module_label: str = "compliance-critical"


class AgentConfig(BaseModel):
    project_name: str
    repo_path: str = "."
    branch_prefix: str = "compliance"
    git_remote: str = "origin"
    runtime: ComplianceRuntimeSettings = Field(default_factory=ComplianceRuntimeSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    junior_llm: JuniorLLMSettings = Field(default_factory=JuniorLLMSettings)
    senior_llm: SeniorLLMSettings = Field(default_factory=SeniorLLMSettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)
    quality_gates: ComplianceQualityGateSettings = Field(default_factory=ComplianceQualityGateSettings)
    escalation: ComplianceEscalationSettings = Field(default_factory=ComplianceEscalationSettings)


def load_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = expand_env_values(raw)
    config = AgentConfig.model_validate(raw)
    if not config.github.repo_name:
        config.github.repo_name = config.project_name
    return config
