"""State definition for the Compliance Checker agent."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # Setup
    config_path: str
    repo_path: str
    dry_run: bool
    config: dict[str, Any]
    repo: dict[str, Any]
    run_id: str
    run_dir: str
    status: str  # "running" | "succeeded" | "failed"

    # Task planning
    task_plan_path: str
    task_plan: dict[str, Any]
    plan_generated: bool
    source_docs_hash: str

    # Current sub-task
    selected_task: dict[str, Any]
    task_id: str
    requirement_id: str
    branch_name: str
    worktree_path: str
    is_critical_module: bool

    # GitHub
    issue: dict[str, Any]

    # LLM execution
    current_tier: str  # "junior" | "senior"
    llm_prompt_path: str
    llm_result: dict[str, Any]
    llm_changed_files: list[str]
    llm_diff_path: str
    llm_diff_stat_path: str

    # Validation gates
    regression_result: dict[str, Any]
    static_analysis_result: dict[str, Any]

    # Commit
    commit_sha: str

    # Failure / escalation
    failure_reason: str
    failure_traceback: str
    summary: str
    escalated: bool
    escalation_reason: str

    # Retry
    retry_count: int

    # Resume
    last_completed_node: str
    resumed_from: str
