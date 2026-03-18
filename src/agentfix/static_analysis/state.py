"""State definition for the Static Analysis Fixer agent."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # Setup
    config_path: str
    repo_path: str
    dry_run: bool
    config: dict
    repo: dict
    run_id: str
    run_dir: str
    status: str  # "running" | "succeeded" | "failed"

    # Static analysis
    warnings_before: list[dict]
    warnings_before_command: dict
    selected_warning: dict
    branch_name: str
    worktree_path: str

    # GitHub issue
    issue: dict

    # LLM task
    llm_task: dict
    llm_result: dict
    llm_changed_files: list[str]
    llm_diff_path: str
    llm_diff_stat_path: str

    # Validation
    build_results: list[dict]
    test_results: list[dict]
    warnings_after: list[dict]
    warnings_after_command: dict

    # Commit
    commit_sha: str

    # Failure / escalation
    failure_reason: str
    failure_traceback: str
    summary: str
    escalated: bool
    escalation_reason: str

    # Resume support
    last_completed_node: str
    resumed_from: str
