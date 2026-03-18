"""State definition for the Test Generator agent."""

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
    status: str

    # Module scanning and selection
    all_modules: list[dict]
    coverage_before: list[dict]
    selected_module: dict
    branch_name: str
    worktree_path: str

    # GitHub issue
    issue: dict

    # Junior LLM task
    llm_task: dict
    llm_result: dict
    llm_changed_files: list[str]
    llm_diff_path: str
    llm_diff_stat_path: str

    # Validation
    compile_result: dict
    test_run_result: dict
    coverage_after: list[dict]
    coverage_delta: float

    # Commit
    commit_sha: str

    # Failure / escalation
    failure_reason: str
    failure_traceback: str
    summary: str
    escalated: bool
    escalation_reason: str

    # Two-tier retry: junior -> senior
    current_tier: str          # "junior" | "senior"
    senior_attempt: int
    senior_prompt_path: str
    senior_result: dict
    senior_changed_files: list[str]
    senior_diff_path: str

    # Quality gates
    regression_result: dict
    static_analysis_result: dict

    # Resume support
    last_completed_node: str
    resumed_from: str
