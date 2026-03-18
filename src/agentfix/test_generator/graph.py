"""LangGraph StateGraph for the Test Generator agent.

Two-tier retry pipeline:
  1. Junior (Codex CLI): Generate unit test, compile, run, measure coverage
  2. Senior (Claude CLI): Fix failing tests with expanded context

Pipeline:
  prepare -> scan_modules -> select_module -> create_issue -> create_worktree
  -> prepare_llm_task -> run_junior -> capture_diff -> compile_test -> run_test
  -> measure_coverage -> run_quality_gates -> commit_and_push -> close_issue -> END

On junior failure:
  handle_failure -> evaluate_tier -> prepare_senior_context -> run_senior
  -> capture_senior_diff -> (re-run quality gates) -> commit_and_push -> ...

Terminal nodes: no_modules, escalate_module, handle_failure
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..shared.audit import append_audit_entry
from ..shared.history import ItemHistory
from ..shared.models import GitHubIssueRef, RepoMetadata
from ..shared.tools.git_ops import (
    capture_diff,
    commit_all,
    create_ephemeral_worktree,
    has_changes,
    has_staged_changes,
    list_changed_paths,
    push_branch,
    remove_worktree,
    stage_paths,
)
from ..shared.tools.github_api import GitHubClient
from ..shared.tools.llm_cli import ensure_binary, run_junior_task, run_senior_task
from ..shared.tools.repo import build_branch_name, collect_repo_metadata, discover_git_root
from ..shared.tools.shell import run_command
from ..shared.utils import dump_json, dump_text, ensure_dir, excerpt, utc_run_id
from .config import AgentConfig, load_config
from .state import AgentState


# ---------------------------------------------------------------------------
# Node ordering for resume support
# ---------------------------------------------------------------------------

_NODE_ORDER: list[str] = [
    "prepare",
    "scan_modules",
    "select_module",
    "create_issue",
    "create_worktree",
    "prepare_llm_task",
    "run_junior",
    "capture_diff",
    "compile_test",
    "run_test",
    "measure_coverage",
    "run_quality_gates",
    "commit_and_push",
    "close_issue",
    # Senior tier nodes:
    "evaluate_tier",
    "prepare_senior_context",
    "run_senior",
    "capture_senior_diff",
]

_NODE_RANK: dict[str, int] = {name: i for i, name in enumerate(_NODE_ORDER)}
_ALLOW_ON_FAILED = {"handle_failure", "evaluate_tier"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(state: AgentState) -> AgentConfig:
    return AgentConfig.model_validate(state["config"])


def _repo(state: AgentState) -> RepoMetadata:
    return RepoMetadata.model_validate(state["repo"])


def _issue(state: AgentState) -> GitHubIssueRef:
    return GitHubIssueRef.model_validate(state["issue"])


def _persist_state_snapshot(state: AgentState) -> None:
    run_dir = state.get("run_dir")
    if run_dir:
        dump_json(Path(run_dir) / "state.json", state)


def _should_skip_on_resume(name: str, state: AgentState) -> bool:
    last_done = state.get("last_completed_node", "")
    if not last_done or not state.get("resumed_from"):
        return False
    last_rank = _NODE_RANK.get(last_done, -1)
    this_rank = _NODE_RANK.get(name, -1)
    return this_rank >= 0 and this_rank <= last_rank


def _wrap_node(name: str, handler):
    def wrapped(state: AgentState) -> dict[str, Any]:
        if state.get("status") == "failed" and name not in _ALLOW_ON_FAILED:
            return {}
        if _should_skip_on_resume(name, state):
            return {}
        try:
            updates = handler(state)
        except Exception as exc:  # noqa: BLE001
            updates = {
                "status": "failed",
                "failure_reason": f"{name}: {exc}",
                "failure_traceback": traceback.format_exc(),
            }
        updates["last_completed_node"] = name
        merged = dict(state)
        merged.update(updates)
        _persist_state_snapshot(merged)
        return updates
    return wrapped


def _workspace_root(state: AgentState) -> Path:
    worktree = state.get("worktree_path")
    return Path(worktree) if worktree else Path(_repo(state).root)


def _load_history(config: AgentConfig) -> ItemHistory | None:
    return ItemHistory(config.runtime.history_file) if config.runtime.history_file else None


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_standard(state: AgentState) -> str:
    return "fail" if state.get("status") == "failed" else "ok"


def _route_after_select(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "fail"
    if not state.get("selected_module"):
        return "done"
    if state.get("escalated"):
        return "escalate"
    return "ok"


def _route_after_evaluate_tier(state: AgentState) -> str:
    """After junior failure, decide: retry junior, escalate to senior, or give up."""
    if state.get("escalated"):
        return "escalate"
    tier = state.get("current_tier", "")
    if tier == "senior":
        return "senior"
    return "end"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _prepare_impl(state: AgentState) -> dict[str, Any]:
    config = load_config(state["config_path"])
    repo_root = discover_git_root(config.repo_path)
    repo = collect_repo_metadata(repo_root, config)

    ensure_binary(config.junior_llm.binary)
    if config.senior_llm.enabled:
        ensure_binary(config.senior_llm.binary)

    run_id = utc_run_id()
    run_dir = str(ensure_dir(Path(config.runtime.artifacts_dir) / run_id))

    return {
        "config": config.model_dump(),
        "repo": repo.model_dump(),
        "run_id": run_id,
        "run_dir": run_dir,
        "status": "running",
        "dry_run": config.runtime.dry_run,
    }


def _scan_modules_impl(state: AgentState) -> dict[str, Any]:
    """Scan the source tree for modules (source files) that can have tests generated.

    NOTE: Customize this for your project structure. This example scans for .c files.
    """
    config = _cfg(state)
    repo_root = Path(_repo(state).root)

    modules: list[dict] = []
    for source_dir in config.coverage.source_paths:
        src_path = repo_root / source_dir
        if not src_path.exists():
            continue
        for f in sorted(src_path.rglob("*.c")):
            rel = str(f.relative_to(repo_root))
            name = f.stem
            # Classify tier (customize for your project)
            tier = "pure_logic"  # default; override with your own classification
            modules.append({"name": name, "source_file": rel, "tier": tier})

    return {"all_modules": modules}


def _select_module_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    modules = state.get("all_modules", [])
    history = _load_history(config)

    # Filter out already-tested or exhausted modules
    eligible = []
    for m in modules:
        fp = f"module:{m['name']}"
        if history:
            if history.was_completed(fp) or history.was_escalated(fp):
                continue
            if history.should_skip(fp, config.runtime.max_retry_per_module):
                continue
        eligible.append(m)

    # Sort by tier preference
    tier_order = {t: i for i, t in enumerate(config.selection_policy.prefer_tiers)}
    eligible.sort(key=lambda m: tier_order.get(m.get("tier", ""), 999))

    if not eligible:
        return {"status": "succeeded", "summary": "No eligible modules remaining."}

    selected = eligible[0]
    branch = build_branch_name(config, selected["name"])

    return {
        "selected_module": selected,
        "branch_name": branch,
        "current_tier": "junior",
        "senior_attempt": 0,
    }


def _no_modules_impl(state: AgentState) -> dict[str, Any]:
    return {"status": "succeeded"}


def _create_issue_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    module = state["selected_module"]
    title = f"[Auto-test] Generate unit tests for {module['name']}"

    if state.get("dry_run") or not config.github.enabled:
        return {"issue": GitHubIssueRef(number=0, url="", title=title).model_dump()}

    client = GitHubClient()
    issue = client.create_issue(
        owner=repo.repo_owner, repo=repo.repo_name,
        title=title,
        body=f"Auto-generate unit tests for `{module['source_file']}` (tier: {module['tier']}).",
        labels=config.github.labels,
    )
    return {"issue": issue.model_dump()}


def _create_worktree_impl(state: AgentState) -> dict[str, Any]:
    repo = _repo(state)
    wt_path = Path(state["run_dir"]) / "worktree"
    create_ephemeral_worktree(Path(repo.root), state["branch_name"], wt_path, repo.current_branch)
    return {"worktree_path": str(wt_path)}


def _prepare_llm_task_impl(state: AgentState) -> dict[str, Any]:
    module = state["selected_module"]
    worktree = _workspace_root(state)

    # Read the source file
    src_path = worktree / module["source_file"]
    src_content = src_path.read_text(encoding="utf-8", errors="replace") if src_path.exists() else ""

    prompt = (
        f"# Generate unit tests\n\n"
        f"## Module: {module['name']}\n"
        f"## Source file: {module['source_file']}\n\n"
        f"```\n{src_content}\n```\n\n"
        f"## Instructions\n"
        f"Generate comprehensive unit tests for this module.\n"
        f"- Use the project's existing test framework and conventions.\n"
        f"- Create mocks/stubs if needed for external dependencies.\n"
        f"- Include both positive and negative test cases.\n"
        f"- Place test files in the project's test directory.\n"
    )

    run_dir = Path(state["run_dir"])
    prompt_path = run_dir / "junior-prompt.md"
    dump_text(prompt_path, prompt)

    return {"llm_task": {"prompt_path": str(prompt_path), "worktree_path": str(worktree)}}


def _run_junior_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state["llm_task"]
    worktree = Path(task["worktree_path"])
    prompt_path = Path(task["prompt_path"])

    result = run_junior_task(worktree, config.junior_llm, prompt_path)

    if not result.ok():
        return {"status": "failed", "failure_reason": f"Junior LLM failed: {excerpt(result.stderr, 500)}", "llm_result": result.model_dump()}

    if not has_changes(worktree):
        return {"status": "failed", "failure_reason": "Junior LLM produced no changes.", "llm_result": result.model_dump()}

    return {"llm_result": result.model_dump()}


def _capture_diff_impl(state: AgentState) -> dict[str, Any]:
    worktree = _workspace_root(state)
    changed = list_changed_paths(worktree)
    if not changed:
        return {"status": "failed", "failure_reason": "No modified files after LLM execution."}

    diff_text, stat_text = capture_diff(worktree)
    run_dir = Path(state["run_dir"])
    dump_text(run_dir / "changes.diff", diff_text)
    dump_text(run_dir / "changes.stat", stat_text)

    return {"llm_changed_files": changed, "llm_diff_path": str(run_dir / "changes.diff")}


def _compile_test_impl(state: AgentState) -> dict[str, Any]:
    """Compile the generated test. Customize the compile command for your project."""
    config = _cfg(state)
    worktree = _workspace_root(state)
    module = state["selected_module"]

    # This is a placeholder — adapt to your build system
    test_dir = worktree / "tests" / "unit" / module["name"]
    if not test_dir.exists():
        return {"status": "failed", "failure_reason": f"Test directory not created: {test_dir}"}

    # Simple gcc compile check
    test_files = list(test_dir.glob("test_*.c"))
    if not test_files:
        return {"status": "failed", "failure_reason": "No test files found."}

    import shlex as _shlex
    flags = " ".join(_shlex.quote(f) for f in config.coverage.compiler_flags)
    cov_flags = " ".join(_shlex.quote(f) for f in config.coverage.coverage_flags)
    result = run_command(
        f"{_shlex.quote(config.coverage.compiler)} {flags} {cov_flags} -o /dev/null {_shlex.quote(str(test_files[0]))} 2>&1",
        cwd=worktree, timeout=120,
    )

    return {"compile_result": result.model_dump()} if result.ok() else {
        "status": "failed",
        "failure_reason": f"Compilation failed: {excerpt(result.stderr, 500)}",
        "compile_result": result.model_dump(),
    }


def _run_test_impl(state: AgentState) -> dict[str, Any]:
    """Run the generated test. Customize for your test runner."""
    worktree = _workspace_root(state)
    result = run_command("bash tests/run_all_tests.sh", cwd=worktree, timeout=300)

    return {"test_run_result": result.model_dump()} if result.ok() else {
        "status": "failed",
        "failure_reason": f"Tests failed: {excerpt(result.stderr, 500)}",
        "test_run_result": result.model_dump(),
    }


def _measure_coverage_impl(state: AgentState) -> dict[str, Any]:
    """Measure code coverage. Returns coverage_delta."""
    # Placeholder: compute coverage delta
    # In a real implementation, use gcov/llvm-cov to measure line coverage
    return {"coverage_delta": 0.0, "coverage_after": []}


def _run_quality_gates_impl(state: AgentState) -> dict[str, Any]:
    """Run regression tests and static analysis on the changes."""
    config = _cfg(state)
    worktree = _workspace_root(state)

    if config.quality_gates.run_regression:
        result = run_command(
            f"bash {config.quality_gates.regression_script}",
            cwd=worktree, timeout=config.quality_gates.regression_timeout_seconds,
        )
        if not result.ok():
            return {"status": "failed", "failure_reason": f"Regression tests failed.", "regression_result": result.model_dump()}

    return {"regression_result": {}, "static_analysis_result": {}}


def _commit_and_push_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    module = state["selected_module"]
    worktree = _workspace_root(state)

    if state.get("dry_run"):
        return {"summary": "Dry-run: commit/push skipped."}

    changed = [f for f in state.get("llm_changed_files", []) if f.strip()]
    if not changed:
        return {"status": "failed", "failure_reason": "No changed files to commit."}

    stage_paths(worktree, changed)
    message = f"test(unit): add tests for {module['name']}"
    sha = commit_all(worktree, message)

    if config.runtime.push_after_commit:
        push_branch(worktree, config.git_remote, state["branch_name"])

    return {"commit_sha": sha}


def _close_issue_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    module = state["selected_module"]

    history = _load_history(config)
    if history:
        history.record(
            fingerprint=f"module:{module['name']}",
            run_id=state.get("run_id", ""),
            outcome="completed",
            tier=state.get("current_tier", "junior"),
        )

    if not state.get("dry_run") and config.github.enabled:
        issue = _issue(state)
        if issue.number > 0:
            client = GitHubClient()
            client.comment_issue(
                repo.repo_owner, repo.repo_name, issue.number,
                f"Tests generated and committed: {state.get('commit_sha', 'N/A')}",
            )
            if config.runtime.close_issue_after_push:
                client.close_issue(repo.repo_owner, repo.repo_name, issue.number)

    return {"status": "succeeded"}


def _handle_failure_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    module = state.get("selected_module", {})
    reason = state.get("failure_reason", "Unknown failure")

    if module:
        history = _load_history(config)
        if history:
            history.record(
                fingerprint=f"module:{module.get('name', '')}",
                run_id=state.get("run_id", ""),
                outcome="failed",
                failure_reason=reason,
                tier=state.get("current_tier", "junior"),
            )

    # Cleanup worktree
    wt = state.get("worktree_path")
    repo = state.get("repo")
    if wt and repo:
        try:
            remove_worktree(Path(repo["root"]), Path(wt))
        except Exception:
            pass

    return {}


def _evaluate_tier_impl(state: AgentState) -> dict[str, Any]:
    """After junior failure, decide whether to escalate to senior or give up."""
    config = _cfg(state)
    module = state.get("selected_module", {})
    if not module:
        return {"escalated": False}

    fp = f"module:{module.get('name', '')}"
    history = _load_history(config)

    junior_fails = history.junior_failure_count(fp) if history else 0
    senior_fails = history.senior_failure_count(fp) if history else 0

    if junior_fails < config.escalation.max_junior_attempts:
        return {"current_tier": "junior", "status": "running", "failure_reason": ""}

    if config.senior_llm.enabled and senior_fails < config.escalation.max_senior_attempts:
        return {"current_tier": "senior", "status": "running", "failure_reason": ""}

    return {
        "escalated": True,
        "escalation_reason": f"Exhausted retries (junior: {junior_fails}, senior: {senior_fails}).",
    }


def _prepare_senior_context_impl(state: AgentState) -> dict[str, Any]:
    """Build an enriched prompt for the senior-tier LLM with failure context."""
    config = _cfg(state)
    module = state["selected_module"]
    worktree = _workspace_root(state)

    # Gather failure history
    history = _load_history(config)
    failures = history.last_failure_details(f"module:{module['name']}") if history else []
    failure_text = "\n".join(
        f"- {f.get('tier', '?')}: {f.get('failure_reason', '')[:200]}"
        for f in failures[-3:]
    )

    src_path = worktree / module["source_file"]
    src_content = src_path.read_text(encoding="utf-8", errors="replace") if src_path.exists() else ""

    prompt = (
        f"# Fix failing unit tests (Senior review)\n\n"
        f"## Module: {module['name']}\n"
        f"## Source: {module['source_file']}\n\n"
        f"```c\n{src_content}\n```\n\n"
        f"## Previous failures\n{failure_text}\n\n"
        f"## Instructions\n"
        f"Fix the unit tests so they compile, run, and pass.\n"
        f"Address the failure reasons listed above.\n"
    )

    run_dir = Path(state["run_dir"])
    prompt_path = run_dir / "senior-prompt.md"
    dump_text(prompt_path, prompt)

    return {"senior_prompt_path": str(prompt_path)}


def _run_senior_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    worktree = _workspace_root(state)
    prompt_path = Path(state["senior_prompt_path"])

    # Clean worktree before senior attempt
    run_command("git checkout -- .", cwd=worktree, timeout=30)

    result = run_senior_task(worktree, config.senior_llm, prompt_path)

    if not result.ok():
        return {"status": "failed", "failure_reason": f"Senior LLM failed: {excerpt(result.stderr, 500)}", "senior_result": result.model_dump()}

    if not has_changes(worktree):
        return {"status": "failed", "failure_reason": "Senior LLM produced no changes.", "senior_result": result.model_dump()}

    return {"senior_result": result.model_dump(), "current_tier": "senior"}


def _capture_senior_diff_impl(state: AgentState) -> dict[str, Any]:
    worktree = _workspace_root(state)
    changed = list_changed_paths(worktree)
    if not changed:
        return {"status": "failed", "failure_reason": "No files changed by senior LLM."}

    diff_text, _ = capture_diff(worktree)
    run_dir = Path(state["run_dir"])
    dump_text(run_dir / "senior.diff", diff_text)

    return {"llm_changed_files": changed, "senior_diff_path": str(run_dir / "senior.diff")}


def _escalate_module_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    module = state.get("selected_module", {})
    reason = state.get("escalation_reason", "")

    if not state.get("dry_run") and config.github.enabled:
        client = GitHubClient()
        client.create_issue(
            owner=repo.repo_owner, repo=repo.repo_name,
            title=f"[Unfixable] Unit test generation failed: {module.get('name', '?')}",
            body=f"## Could not auto-generate tests\n\nReason: {reason}\n\nNeeds manual test creation.",
            labels=config.github.labels + [config.escalation.escalation_label],
        )

    history = _load_history(config)
    if history:
        history.record(
            fingerprint=f"module:{module.get('name', '')}",
            run_id=state.get("run_id", ""),
            outcome="escalated",
            failure_reason=reason,
        )

    return {"status": "succeeded"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("prepare", _wrap_node("prepare", _prepare_impl))
    graph.add_node("scan_modules", _wrap_node("scan_modules", _scan_modules_impl))
    graph.add_node("select_module", _wrap_node("select_module", _select_module_impl))
    graph.add_node("no_modules", _wrap_node("no_modules", _no_modules_impl))
    graph.add_node("escalate_module", _wrap_node("escalate_module", _escalate_module_impl))
    graph.add_node("create_issue", _wrap_node("create_issue", _create_issue_impl))
    graph.add_node("create_worktree", _wrap_node("create_worktree", _create_worktree_impl))
    graph.add_node("prepare_llm_task", _wrap_node("prepare_llm_task", _prepare_llm_task_impl))
    graph.add_node("run_junior", _wrap_node("run_junior", _run_junior_impl))
    graph.add_node("capture_diff", _wrap_node("capture_diff", _capture_diff_impl))
    graph.add_node("compile_test", _wrap_node("compile_test", _compile_test_impl))
    graph.add_node("run_test", _wrap_node("run_test", _run_test_impl))
    graph.add_node("measure_coverage", _wrap_node("measure_coverage", _measure_coverage_impl))
    graph.add_node("run_quality_gates", _wrap_node("run_quality_gates", _run_quality_gates_impl))
    graph.add_node("commit_and_push", _wrap_node("commit_and_push", _commit_and_push_impl))
    graph.add_node("close_issue", _wrap_node("close_issue", _close_issue_impl))
    graph.add_node("handle_failure", _wrap_node("handle_failure", _handle_failure_impl))
    graph.add_node("evaluate_tier", _wrap_node("evaluate_tier", _evaluate_tier_impl))
    graph.add_node("prepare_senior_context", _wrap_node("prepare_senior_context", _prepare_senior_context_impl))
    graph.add_node("run_senior", _wrap_node("run_senior", _run_senior_impl))
    graph.add_node("capture_senior_diff", _wrap_node("capture_senior_diff", _capture_senior_diff_impl))

    # --- Edges: Main pipeline ---
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", _route_standard, {"ok": "scan_modules", "fail": "handle_failure"})
    graph.add_conditional_edges("scan_modules", _route_standard, {"ok": "select_module", "fail": "handle_failure"})
    graph.add_conditional_edges("select_module", _route_after_select, {
        "ok": "create_issue", "done": "no_modules", "escalate": "escalate_module", "fail": "handle_failure",
    })
    graph.add_edge("no_modules", END)
    graph.add_edge("escalate_module", END)

    graph.add_conditional_edges("create_issue", _route_standard, {"ok": "create_worktree", "fail": "handle_failure"})
    graph.add_conditional_edges("create_worktree", _route_standard, {"ok": "prepare_llm_task", "fail": "handle_failure"})
    graph.add_conditional_edges("prepare_llm_task", _route_standard, {"ok": "run_junior", "fail": "handle_failure"})
    graph.add_conditional_edges("run_junior", _route_standard, {"ok": "capture_diff", "fail": "handle_failure"})
    graph.add_conditional_edges("capture_diff", _route_standard, {"ok": "compile_test", "fail": "handle_failure"})
    graph.add_conditional_edges("compile_test", _route_standard, {"ok": "run_test", "fail": "handle_failure"})
    graph.add_conditional_edges("run_test", _route_standard, {"ok": "measure_coverage", "fail": "handle_failure"})
    graph.add_conditional_edges("measure_coverage", _route_standard, {"ok": "run_quality_gates", "fail": "handle_failure"})
    graph.add_conditional_edges("run_quality_gates", _route_standard, {"ok": "commit_and_push", "fail": "handle_failure"})
    graph.add_conditional_edges("commit_and_push", _route_standard, {"ok": "close_issue", "fail": "handle_failure"})
    graph.add_edge("close_issue", END)

    # --- Failure -> Tier evaluation -> Senior retry ---
    graph.add_edge("handle_failure", "evaluate_tier")
    graph.add_conditional_edges("evaluate_tier", _route_after_evaluate_tier, {
        "senior": "prepare_senior_context",
        "escalate": "escalate_module",
        "end": END,
    })

    graph.add_conditional_edges("prepare_senior_context", _route_standard, {"ok": "run_senior", "fail": END})
    graph.add_conditional_edges("run_senior", _route_standard, {"ok": "capture_senior_diff", "fail": "handle_failure"})
    graph.add_conditional_edges("capture_senior_diff", _route_standard, {"ok": "compile_test", "fail": "handle_failure"})

    return graph.compile()
