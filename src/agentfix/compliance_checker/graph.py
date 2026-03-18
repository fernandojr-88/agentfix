"""LangGraph StateGraph for the Compliance Checker agent.

This agent reads compliance/regulatory requirement documents, generates a task plan
via LLM, then implements each sub-task using a two-tier (junior/senior) approach.

Key features:
  - LLM-generated task plan from requirement documents
  - Task plan persistence and change detection (via source doc hash)
  - Critical module detection -> routes to senior tier directly
  - After commit: critical modules get labeled for human review instead of auto-merging
  - In-graph retry loop: handle_failure -> evaluate_tier -> retry or escalate

Pipeline:
  prepare -> load_task_plan -> [generate_task_plan -> persist_task_plan] -> select_next_task
  -> create_issue -> create_worktree -> prepare_llm_task -> run_junior|run_senior
  -> capture_diff -> run_regression -> run_static_analysis -> commit_and_push
  -> close_issue (normal) | label_for_review (critical) -> END

Retry loop:
  handle_failure -> evaluate_tier -> prepare_llm_task -> run_junior|run_senior -> ...
"""

from __future__ import annotations

import hashlib
import traceback
from pathlib import Path
from typing import Any

import yaml
from langgraph.graph import END, START, StateGraph

from ..shared.audit import append_audit_entry
from ..shared.history import ItemHistory
from ..shared.models import GitHubIssueRef, RepoMetadata
from ..shared.tools.git_ops import (
    capture_diff,
    commit_all,
    create_ephemeral_worktree,
    has_changes,
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
# Node ordering for resume
# ---------------------------------------------------------------------------

_NODE_ORDER: list[str] = [
    "prepare", "load_task_plan", "generate_task_plan", "persist_task_plan",
    "select_next_task", "create_issue", "create_worktree", "prepare_llm_task",
    "run_junior", "run_senior", "capture_diff", "run_regression", "run_static_analysis",
    "commit_and_push", "close_issue", "label_for_review", "evaluate_tier",
]
_NODE_RANK: dict[str, int] = {name: i for i, name in enumerate(_NODE_ORDER)}
_ALLOW_ON_FAILED = {"handle_failure", "evaluate_tier"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(state: AgentState) -> AgentConfig:
    return AgentConfig.model_validate(state["config"])


def _persist_state_snapshot(state: AgentState) -> None:
    run_dir = state.get("run_dir")
    if run_dir:
        dump_json(Path(run_dir) / "state.json", state)


def _should_skip_on_resume(name: str, state: AgentState) -> bool:
    last_done = state.get("last_completed_node", "")
    if not last_done or not state.get("resumed_from"):
        return False
    return _NODE_RANK.get(name, -1) <= _NODE_RANK.get(last_done, -1)


def _wrap_node(name: str, handler):
    def wrapped(state: AgentState) -> dict[str, Any]:
        if state.get("status") == "failed" and name not in _ALLOW_ON_FAILED:
            return {}
        if _should_skip_on_resume(name, state):
            return {}
        try:
            updates = handler(state)
        except Exception as exc:  # noqa: BLE001
            updates = {"status": "failed", "failure_reason": f"{name}: {exc}", "failure_traceback": traceback.format_exc()}
        updates["last_completed_node"] = name
        merged = dict(state)
        merged.update(updates)
        _persist_state_snapshot(merged)
        return updates
    return wrapped


def _cleanup_worktree(state: AgentState) -> None:
    wt = state.get("worktree_path")
    repo = state.get("repo")
    if wt and repo:
        try:
            remove_worktree(Path(repo["root"]), Path(wt))
        except Exception:
            pass


def _load_history(config: AgentConfig) -> ItemHistory:
    path = config.runtime.history_file or str(Path(config.runtime.artifacts_dir) / "history.json")
    return ItemHistory(path)


def _is_critical_module(files: list[str], critical_modules: list[str]) -> bool:
    """Check if any of the files belong to a critical module."""
    for f in files:
        stem = Path(f).stem
        if stem in critical_modules:
            return True
    return False


def _compute_docs_hash(docs_root: Path) -> str:
    """SHA256 hash of all requirement docs for change detection."""
    content = ""
    if docs_root.exists():
        for f in sorted(docs_root.rglob("*.md")):
            content += f.read_text(encoding="utf-8", errors="replace")
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_standard(state: AgentState) -> str:
    return "fail" if state.get("status") == "failed" else "ok"

def _route_after_load_plan(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "fail"
    return "has_plan" if state.get("task_plan") else "no_plan"

def _route_after_select(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "fail"
    if not state.get("selected_task"):
        return "done"
    if state.get("escalated"):
        return "escalate"
    return "ok"

def _route_after_prepare_llm(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "fail"
    if state.get("is_critical_module") or state.get("current_tier") == "senior":
        return "senior"
    return "junior"

def _route_after_commit(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "fail"
    return "critical" if state.get("is_critical_module") else "normal"

def _route_after_evaluate_tier(state: AgentState) -> str:
    if state.get("escalated"):
        return "escalate"
    tier = state.get("current_tier", "")
    if tier in ("junior", "senior"):
        return "retry"
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
        "config": config.model_dump(), "repo": repo.model_dump(),
        "run_id": run_id, "run_dir": run_dir,
        "status": "running", "dry_run": config.runtime.dry_run,
    }


def _load_task_plan_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    plan_path = config.runtime.task_plan_file or str(Path(config.runtime.artifacts_dir) / "task_plan.yaml")

    plan_file = Path(plan_path)
    if not plan_file.exists():
        return {"task_plan_path": plan_path, "task_plan": {}, "plan_generated": False}

    plan = yaml.safe_load(plan_file.read_text(encoding="utf-8"))

    # Check if source docs changed
    repo_root = Path(state["repo"]["root"])
    docs_root = repo_root / config.compliance.docs_root
    current_hash = _compute_docs_hash(docs_root)
    if current_hash != plan.get("source_hash", ""):
        return {"task_plan_path": plan_path, "task_plan": {}, "plan_generated": False}

    return {"task_plan_path": plan_path, "task_plan": plan, "plan_generated": False}


def _generate_task_plan_impl(state: AgentState) -> dict[str, Any]:
    """Generate a task plan from compliance docs using the senior LLM."""
    config = _cfg(state)
    repo_root = Path(state["repo"]["root"])
    docs_root = repo_root / config.compliance.docs_root
    run_dir = Path(state["run_dir"])

    # Read all requirement docs
    doc_content = ""
    if docs_root.exists():
        for f in sorted(docs_root.rglob("*.md")):
            doc_content += f"\n\n## {f.name}\n\n{f.read_text(encoding='utf-8', errors='replace')}"

    prompt = (
        f"# Generate compliance task plan\n\n"
        f"Read the following compliance/regulatory requirement documents and generate\n"
        f"a YAML task plan with sub-tasks that need to be implemented.\n\n"
        f"Skip requirements: {config.compliance.skip_requirements}\n"
        f"Critical modules: {config.compliance.critical_modules}\n\n"
        f"## Requirement documents\n{doc_content}\n\n"
        f"## Output format\n"
        f"Return a YAML document with this structure:\n"
        f"```yaml\n"
        f"tasks:\n"
        f"  - id: \"REQ-1.1-description\"\n"
        f"    requirement: \"1.1\"\n"
        f"    title: \"Short title\"\n"
        f"    description: \"What needs to be done\"\n"
        f"    type: \"implementation+test\"\n"
        f"    files_involved: [\"src/module.c\"]\n"
        f"    priority: 1\n"
        f"    status: \"pending\"\n"
        f"```\n"
    )

    prompt_path = run_dir / "planning-prompt.md"
    dump_text(prompt_path, prompt)

    result = run_senior_task(repo_root, config.senior_llm, prompt_path)
    if not result.ok():
        return {"status": "failed", "failure_reason": f"Planning LLM failed: {excerpt(result.stderr, 500)}"}

    # Parse YAML from output
    raw_output = result.stdout.strip()
    yaml_text = raw_output
    if "```yaml" in yaml_text:
        start = yaml_text.index("```yaml") + 7
        end = yaml_text.index("```", start)
        yaml_text = yaml_text[start:end].strip()

    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return {"status": "failed", "failure_reason": f"Failed to parse planning YAML: {exc}"}

    if not isinstance(parsed, dict) or "tasks" not in parsed:
        return {"status": "failed", "failure_reason": "Planning output missing 'tasks' key."}

    source_hash = _compute_docs_hash(repo_root / config.compliance.docs_root)
    plan = {"source_hash": source_hash, "tasks": parsed["tasks"], "generated_at": utc_run_id()}

    return {"task_plan": plan, "plan_generated": True, "source_docs_hash": source_hash}


def _persist_task_plan_impl(state: AgentState) -> dict[str, Any]:
    plan_path = state["task_plan_path"]
    Path(plan_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plan_path).write_text(yaml.dump(state["task_plan"], allow_unicode=True), encoding="utf-8")
    return {}


def _select_next_task_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    plan = state["task_plan"]
    tasks = plan.get("tasks", [])
    history = _load_history(config)
    skip_reqs = set(config.compliance.skip_requirements)

    # Find next pending task
    for task in tasks:
        if task.get("status") != "pending":
            continue
        if task.get("requirement", "") in skip_reqs:
            continue
        fp = f"task:{task['id']}"
        if history.was_completed(fp) or history.was_escalated(fp):
            continue

        # Check criticality
        files = task.get("files_involved", [])
        critical = _is_critical_module(files, config.compliance.critical_modules)

        # Determine tier
        junior_fails = history.junior_failure_count(fp)
        senior_fails = history.senior_failure_count(fp)
        if junior_fails + senior_fails >= config.escalation.max_junior_attempts + config.escalation.max_senior_attempts:
            return {
                "selected_task": task, "task_id": task["id"],
                "escalated": True, "escalation_reason": "Both tiers exhausted.",
            }

        tier = "senior" if critical or junior_fails >= config.escalation.max_junior_attempts else "junior"
        branch = build_branch_name(config, task["id"])

        return {
            "selected_task": task, "task_id": task["id"],
            "requirement_id": task.get("requirement", ""),
            "branch_name": branch, "is_critical_module": critical,
            "current_tier": tier, "escalated": False, "retry_count": 0,
        }

    return {"selected_task": {}, "status": "succeeded", "summary": "No pending tasks remaining."}


def _create_issue_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state["selected_task"]

    if state.get("dry_run") or not config.github.enabled:
        return {"issue": GitHubIssueRef(number=0, url="", title=task.get("title", "")).model_dump()}

    repo = state["repo"]
    labels = list(config.github.labels)
    if state.get("is_critical_module"):
        labels.append(config.escalation.critical_module_label)

    client = GitHubClient()
    issue = client.create_issue(
        owner=repo["repo_owner"], repo=repo["repo_name"],
        title=f"[Compliance] {task.get('title', task['id'])}",
        body=f"## Requirement: {task.get('requirement', 'N/A')}\n\n{task.get('description', '')}",
        labels=labels,
    )
    return {"issue": issue.model_dump()}


def _create_worktree_impl(state: AgentState) -> dict[str, Any]:
    repo_root = Path(state["repo"]["root"])
    wt_path = Path(state["run_dir"]) / "worktree"
    create_ephemeral_worktree(repo_root, state["branch_name"], wt_path, "HEAD")
    return {"worktree_path": str(wt_path)}


def _prepare_llm_task_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state["selected_task"]
    wt = Path(state["worktree_path"])
    run_dir = Path(state["run_dir"])
    tier = state.get("current_tier", "junior")

    # Read source files for context
    file_context = ""
    for f in task.get("files_involved", []):
        fp = wt / f
        if fp.exists():
            file_context += f"\n### {f}\n```\n{fp.read_text(encoding='utf-8', errors='replace')[:4000]}\n```\n"

    prompt = (
        f"# {'[Senior Review] ' if tier == 'senior' else ''}Implement compliance task\n\n"
        f"## Task: {task.get('title', task['id'])}\n"
        f"## Requirement: {task.get('requirement', 'N/A')}\n"
        f"## Description: {task.get('description', '')}\n"
        f"## Type: {task.get('type', 'implementation')}\n\n"
        f"## Files involved\n{file_context}\n\n"
        f"## Instructions\n"
        f"Implement the changes required by this compliance requirement.\n"
        f"Make minimal, targeted changes. Include tests where appropriate.\n"
    )

    prompt_path = run_dir / f"{tier}-prompt.md"
    dump_text(prompt_path, prompt)
    return {"llm_prompt_path": str(prompt_path)}


def _run_junior_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    wt = Path(state["worktree_path"])
    run_command("git checkout -- .", cwd=wt, timeout=30)
    result = run_junior_task(wt, config.junior_llm, Path(state["llm_prompt_path"]))
    if not result.ok():
        return {"status": "failed", "failure_reason": f"Junior failed: {excerpt(result.stderr, 500)}", "llm_result": result.model_dump()}
    if not has_changes(wt):
        return {"status": "failed", "failure_reason": "Junior produced no changes.", "llm_result": result.model_dump()}
    return {"llm_result": result.model_dump()}


def _run_senior_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    wt = Path(state["worktree_path"])
    run_command("git checkout -- .", cwd=wt, timeout=30)
    result = run_senior_task(wt, config.senior_llm, Path(state["llm_prompt_path"]))
    if not result.ok():
        return {"status": "failed", "failure_reason": f"Senior failed: {excerpt(result.stderr, 500)}", "llm_result": result.model_dump()}
    if not has_changes(wt):
        return {"status": "failed", "failure_reason": "Senior produced no changes.", "llm_result": result.model_dump()}
    return {"llm_result": result.model_dump(), "current_tier": "senior"}


def _capture_diff_impl(state: AgentState) -> dict[str, Any]:
    wt = Path(state["worktree_path"])
    diff_text, stat_text = capture_diff(wt)
    changed = list_changed_paths(wt)
    run_dir = Path(state["run_dir"])
    dump_text(run_dir / "changes.diff", diff_text)
    return {"llm_changed_files": changed, "llm_diff_path": str(run_dir / "changes.diff")}


def _run_regression_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    if not config.quality_gates.run_regression:
        return {"regression_result": {}}
    wt = Path(state["worktree_path"])
    result = run_command(f"bash {config.quality_gates.regression_script}", cwd=wt, timeout=config.quality_gates.regression_timeout_seconds)
    if not result.ok():
        return {"status": "failed", "failure_reason": "Regression tests failed.", "regression_result": result.model_dump()}
    return {"regression_result": {"returncode": 0}}


def _run_static_analysis_impl(state: AgentState) -> dict[str, Any]:
    """Run static analysis on changed files as a quality gate."""
    config = _cfg(state)
    if not config.quality_gates.run_static_analysis or not config.quality_gates.static_analysis_command:
        return {"static_analysis_result": {}}
    wt = Path(state["worktree_path"])
    changed = state.get("llm_changed_files", [])
    if not changed:
        return {"static_analysis_result": {}}
    import shlex as _shlex
    files_str = " ".join(_shlex.quote(f) for f in changed)
    args = " ".join(_shlex.quote(a) for a in config.quality_gates.static_analysis_args)
    command = f"{config.quality_gates.static_analysis_command} {args} {files_str}"
    result = run_command(command, cwd=wt, timeout=300)
    if result.returncode != 0:
        return {"status": "failed", "failure_reason": f"Static analysis errors: {excerpt(result.stderr, 500)}", "static_analysis_result": result.model_dump()}
    return {"static_analysis_result": result.model_dump()}


def _commit_and_push_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state["selected_task"]
    wt = Path(state["worktree_path"])
    changed = [f for f in state.get("llm_changed_files", []) if not f.endswith((".o", ".gcno", ".gcda"))]
    if not changed:
        return {"status": "failed", "failure_reason": "No files to commit."}
    stage_paths(wt, changed)
    message = f"compliance({task.get('requirement', 'N/A')}): {task.get('title', task['id'])}"
    sha = commit_all(wt, message)
    if config.runtime.push_after_commit and not state.get("dry_run"):
        push_branch(wt, config.git_remote, state["branch_name"])
    return {"commit_sha": sha}


def _close_issue_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state["selected_task"]
    history = _load_history(config)
    history.record(fingerprint=f"task:{task['id']}", run_id=state["run_id"], outcome="completed", tier=state.get("current_tier", "junior"))

    if not state.get("dry_run") and config.github.enabled:
        issue_data = state.get("issue", {})
        if issue_data.get("number", 0) > 0:
            repo = state["repo"]
            client = GitHubClient()
            client.comment_issue(repo["repo_owner"], repo["repo_name"], issue_data["number"], f"Completed: {state.get('commit_sha', 'N/A')}")
            if config.runtime.close_issue_after_push:
                client.close_issue(repo["repo_owner"], repo["repo_name"], issue_data["number"])

    # Update task plan
    plan_path = state.get("task_plan_path", "")
    if plan_path and Path(plan_path).exists():
        plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
        for t in plan.get("tasks", []):
            if t.get("id") == task["id"]:
                t["status"] = "completed"
        Path(plan_path).write_text(yaml.dump(plan, allow_unicode=True), encoding="utf-8")

    return {"status": "succeeded", "summary": f"Task {task['id']} completed."}


def _label_for_review_impl(state: AgentState) -> dict[str, Any]:
    """For critical modules: leave branch open for human review."""
    config = _cfg(state)
    if not state.get("dry_run") and config.github.enabled:
        issue_data = state.get("issue", {})
        if issue_data.get("number", 0) > 0:
            repo = state["repo"]
            client = GitHubClient()
            client.comment_issue(
                repo["repo_owner"], repo["repo_name"], issue_data["number"],
                f"Completed ({state.get('commit_sha', 'N/A')}). Branch left open for human review (critical module).",
            )
            client.add_label(repo["repo_owner"], repo["repo_name"], issue_data["number"], config.escalation.critical_module_label)

    history = _load_history(config)
    task = state["selected_task"]
    history.record(fingerprint=f"task:{task['id']}", run_id=state["run_id"], outcome="completed", tier="senior")

    return {"status": "succeeded", "summary": f"Task {task['id']} completed (awaiting human review)."}


def _handle_failure_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state.get("selected_task", {})
    if not task:
        return {}
    history = _load_history(config)
    history.record(
        fingerprint=f"task:{task['id']}", run_id=state["run_id"],
        outcome="failed", failure_reason=state.get("failure_reason", ""),
        tier=state.get("current_tier", "junior"),
    )
    return {}


def _evaluate_tier_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state.get("selected_task", {})
    if not task:
        _cleanup_worktree(state)
        return {"escalated": False}

    retry_count = state.get("retry_count", 0) + 1
    if retry_count > 2:
        _cleanup_worktree(state)
        return {"escalated": True, "escalation_reason": "Max retries in single run.", "current_tier": "", "retry_count": retry_count}

    fp = f"task:{task['id']}"
    history = _load_history(config)
    junior_fails = history.junior_failure_count(fp)
    senior_fails = history.senior_failure_count(fp)

    if junior_fails < config.escalation.max_junior_attempts:
        return {"current_tier": "junior", "status": "running", "failure_reason": "", "retry_count": retry_count}
    if config.senior_llm.enabled and senior_fails < config.escalation.max_senior_attempts:
        return {"current_tier": "senior", "status": "running", "failure_reason": "", "retry_count": retry_count}

    _cleanup_worktree(state)
    return {"escalated": True, "escalation_reason": "Both tiers exhausted.", "current_tier": "", "retry_count": retry_count}


def _escalate_task_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = state.get("selected_task", {})
    _cleanup_worktree(state)

    history = _load_history(config)
    history.record(fingerprint=f"task:{task['id']}", run_id=state["run_id"], outcome="escalated", failure_reason=state.get("escalation_reason", ""))

    if not state.get("dry_run") and config.github.enabled:
        repo = state["repo"]
        client = GitHubClient()
        client.create_issue(
            owner=repo["repo_owner"], repo=repo["repo_name"],
            title=f"[Escalation] {task.get('title', task['id'])}",
            body=f"Could not auto-implement. Reason: {state.get('escalation_reason', 'N/A')}",
            labels=config.github.labels + [config.escalation.escalation_label],
        )

    return {"status": "succeeded", "summary": f"Task {task['id']} escalated."}


def _no_tasks_impl(state: AgentState) -> dict[str, Any]:
    return {"status": "succeeded"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("prepare", _wrap_node("prepare", _prepare_impl))
    graph.add_node("load_task_plan", _wrap_node("load_task_plan", _load_task_plan_impl))
    graph.add_node("generate_task_plan", _wrap_node("generate_task_plan", _generate_task_plan_impl))
    graph.add_node("persist_task_plan", _wrap_node("persist_task_plan", _persist_task_plan_impl))
    graph.add_node("select_next_task", _wrap_node("select_next_task", _select_next_task_impl))
    graph.add_node("create_issue", _wrap_node("create_issue", _create_issue_impl))
    graph.add_node("create_worktree", _wrap_node("create_worktree", _create_worktree_impl))
    graph.add_node("prepare_llm_task", _wrap_node("prepare_llm_task", _prepare_llm_task_impl))
    graph.add_node("run_junior", _wrap_node("run_junior", _run_junior_impl))
    graph.add_node("run_senior", _wrap_node("run_senior", _run_senior_impl))
    graph.add_node("capture_diff", _wrap_node("capture_diff", _capture_diff_impl))
    graph.add_node("run_regression", _wrap_node("run_regression", _run_regression_impl))
    graph.add_node("run_static_analysis", _wrap_node("run_static_analysis", _run_static_analysis_impl))
    graph.add_node("commit_and_push", _wrap_node("commit_and_push", _commit_and_push_impl))
    graph.add_node("close_issue", _wrap_node("close_issue", _close_issue_impl))
    graph.add_node("label_for_review", _wrap_node("label_for_review", _label_for_review_impl))
    graph.add_node("handle_failure", _wrap_node("handle_failure", _handle_failure_impl))
    graph.add_node("evaluate_tier", _wrap_node("evaluate_tier", _evaluate_tier_impl))
    graph.add_node("escalate_task", _wrap_node("escalate_task", _escalate_task_impl))
    graph.add_node("no_tasks_remaining", _wrap_node("no_tasks_remaining", _no_tasks_impl))

    # --- Planning phase ---
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", _route_standard, {"ok": "load_task_plan", "fail": "handle_failure"})
    graph.add_conditional_edges("load_task_plan", _route_after_load_plan, {
        "has_plan": "select_next_task", "no_plan": "generate_task_plan", "fail": "handle_failure",
    })
    graph.add_conditional_edges("generate_task_plan", _route_standard, {"ok": "persist_task_plan", "fail": "handle_failure"})
    graph.add_edge("persist_task_plan", "select_next_task")

    # --- Task selection ---
    graph.add_conditional_edges("select_next_task", _route_after_select, {
        "ok": "create_issue", "done": "no_tasks_remaining", "escalate": "escalate_task", "fail": "handle_failure",
    })
    graph.add_edge("no_tasks_remaining", END)
    graph.add_edge("escalate_task", END)

    # --- Execution pipeline ---
    graph.add_conditional_edges("create_issue", _route_standard, {"ok": "create_worktree", "fail": "handle_failure"})
    graph.add_conditional_edges("create_worktree", _route_standard, {"ok": "prepare_llm_task", "fail": "handle_failure"})
    graph.add_conditional_edges("prepare_llm_task", _route_after_prepare_llm, {
        "junior": "run_junior", "senior": "run_senior", "fail": "handle_failure",
    })

    graph.add_conditional_edges("run_junior", _route_standard, {"ok": "capture_diff", "fail": "handle_failure"})
    graph.add_conditional_edges("run_senior", _route_standard, {"ok": "capture_diff", "fail": "handle_failure"})
    graph.add_conditional_edges("capture_diff", _route_standard, {"ok": "run_regression", "fail": "handle_failure"})
    graph.add_conditional_edges("run_regression", _route_standard, {"ok": "run_static_analysis", "fail": "handle_failure"})
    graph.add_conditional_edges("run_static_analysis", _route_standard, {"ok": "commit_and_push", "fail": "handle_failure"})

    # --- Post-commit routing ---
    graph.add_conditional_edges("commit_and_push", _route_after_commit, {
        "critical": "label_for_review", "normal": "close_issue", "fail": "handle_failure",
    })
    graph.add_edge("close_issue", END)
    graph.add_edge("label_for_review", END)

    # --- Failure -> retry loop ---
    graph.add_edge("handle_failure", "evaluate_tier")
    graph.add_conditional_edges("evaluate_tier", _route_after_evaluate_tier, {
        "retry": "prepare_llm_task", "escalate": "escalate_task", "end": END,
    })

    return graph.compile()
