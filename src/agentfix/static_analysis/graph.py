"""LangGraph StateGraph for the Static Analysis Fixer agent.

Pipeline:
  prepare -> run_analysis_before -> select_warning -> create_issue -> create_worktree
  -> prepare_llm_task -> run_llm -> capture_diff -> build_code -> run_tests
  -> run_analysis_after -> commit_and_push -> close_issue -> END

Terminal nodes: no_warnings, escalate_warning, handle_failure
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..shared.audit import append_audit_entry
from ..shared.history import ItemHistory
from ..shared.models import GitHubIssueRef, LLMTask, RepoMetadata
from ..shared.tools.git_ops import (
    capture_diff,
    commit_all,
    create_ephemeral_worktree,
    ensure_clean_worktree,
    has_changes,
    has_staged_changes,
    list_changed_paths,
    push_branch,
    remove_worktree,
    stage_paths,
)
from ..shared.tools.github_api import GitHubClient
from ..shared.tools.llm_cli import ensure_binary, run_junior_task
from ..shared.tools.repo import build_branch_name, collect_repo_metadata, discover_git_root
from ..shared.utils import dump_json, dump_text, ensure_dir, excerpt, utc_run_id
from .config import AgentConfig, load_config
from .models import WarningCandidate
from .state import AgentState


# ---------------------------------------------------------------------------
# Node ordering for resume support
# ---------------------------------------------------------------------------

_NODE_ORDER: list[str] = [
    "prepare",
    "run_analysis_before",
    "select_warning",
    "create_issue",
    "create_worktree",
    "prepare_llm_task",
    "run_llm",
    "capture_diff",
    "build_code",
    "run_tests",
    "run_analysis_after",
    "commit_and_push",
    "close_issue",
]

_NODE_RANK: dict[str, int] = {name: i for i, name in enumerate(_NODE_ORDER)}


# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------

def _cfg(state: AgentState) -> AgentConfig:
    return AgentConfig.model_validate(state["config"])


def _repo(state: AgentState) -> RepoMetadata:
    return RepoMetadata.model_validate(state["repo"])


def _warning(payload: dict[str, Any]) -> WarningCandidate:
    return WarningCandidate.model_validate(payload)


def _warnings(items: list[dict[str, Any]]) -> list[WarningCandidate]:
    return [WarningCandidate.model_validate(item) for item in items]


def _issue(state: AgentState) -> GitHubIssueRef:
    return GitHubIssueRef.model_validate(state["issue"])


def _persist_state_snapshot(state: AgentState) -> None:
    run_dir = state.get("run_dir")
    if not run_dir:
        return
    dump_json(Path(run_dir) / "state.json", state)


def _should_skip_on_resume(name: str, state: AgentState) -> bool:
    last_done = state.get("last_completed_node", "")
    if not last_done or not state.get("resumed_from"):
        return False
    last_rank = _NODE_RANK.get(last_done, -1)
    this_rank = _NODE_RANK.get(name, -1)
    return this_rank >= 0 and this_rank <= last_rank


def _wrap_node(name: str, handler):
    """Decorator: catch exceptions, skip on resume, persist state after each node."""
    def wrapped(state: AgentState) -> dict[str, Any]:
        if state.get("status") == "failed" and name != "handle_failure":
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
    if worktree:
        return Path(worktree)
    return Path(_repo(state).root)


def _load_history(config: AgentConfig) -> ItemHistory | None:
    if config.runtime.history_file:
        return ItemHistory(config.runtime.history_file)
    return None


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_standard(state: AgentState) -> str:
    return "fail" if state.get("status") == "failed" else "ok"


def _route_after_select(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "fail"
    if not state.get("selected_warning"):
        return "done"
    if state.get("escalated"):
        return "escalate"
    return "ok"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _prepare_impl(state: AgentState) -> dict[str, Any]:
    config = load_config(state["config_path"])
    dry_run = state.get("dry_run", config.runtime.dry_run)
    repo_root = discover_git_root(state.get("repo_path") or config.repo_path)

    if config.runtime.require_clean_worktree:
        ensure_clean_worktree(repo_root)

    ensure_binary(config.junior_llm.binary)
    repo_meta = collect_repo_metadata(repo_root, config)

    artifacts_root = (
        Path(config.runtime.artifacts_dir)
        if config.runtime.artifacts_dir.strip()
        else Path(tempfile.gettempdir()) / "static_analysis_agent" / config.project_name
    )
    run_id = utc_run_id()
    run_dir = ensure_dir(artifacts_root / run_id)

    dump_json(run_dir / "resolved_config.json", config.model_dump())

    return {
        "config": config.model_dump(),
        "repo": repo_meta.model_dump(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "running",
        "dry_run": dry_run,
        "summary": "",
    }


def _run_analysis_before_impl(state: AgentState) -> dict[str, Any]:
    """Run the static analysis tool and collect warnings.

    Configure `static_analysis.command` in your YAML config to point to your
    analysis tool. The tool should produce XML output (cppcheck-compatible)
    or JSON output. Override this function for custom output formats.

    Supported tools (via config): cppcheck, clang-tidy, pylint, ESLint, etc.
    """
    from ..shared.tools.shell import run_command

    config = _cfg(state)
    repo = _repo(state)
    run_dir = Path(state["run_dir"])

    import shlex as _shlex

    if not config.static_analysis.command:
        return {"status": "failed", "failure_reason": "No static_analysis.command configured."}

    source_paths = " ".join(_shlex.quote(p) for p in config.static_analysis.source_paths)
    command = config.static_analysis.command.replace("{source_paths}", source_paths)

    result = run_command(command, cwd=Path(repo.root), timeout=config.runtime.command_timeout_seconds)

    # Parse output into WarningCandidate objects
    warnings: list[dict] = []

    if config.static_analysis.output_format == "xml":
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(result.stdout)
            for error in root.iter("error"):
                for location in error.iter("location"):
                    w = WarningCandidate(
                        id=error.get("id", ""),
                        severity=error.get("severity", ""),
                        file=location.get("file", ""),
                        line=int(location.get("line", "0")),
                        column=int(location.get("column", "0")) if location.get("column") else None,
                        message=error.get("msg", ""),
                        verbose=error.get("verbose", ""),
                        cwe=error.get("cwe", ""),
                    )
                    warnings.append(w.model_dump())
                    break
        except ET.ParseError:
            pass
    elif config.static_analysis.output_format == "json":
        import json as _json
        try:
            items = _json.loads(result.stdout)
            if isinstance(items, list):
                for item in items:
                    w = WarningCandidate(
                        id=str(item.get("id", item.get("rule", item.get("code", "")))),
                        severity=str(item.get("severity", item.get("type", "warning"))),
                        file=str(item.get("file", item.get("path", ""))),
                        line=int(item.get("line", item.get("lineno", 0))),
                        message=str(item.get("message", item.get("msg", ""))),
                    )
                    warnings.append(w.model_dump())
        except (ValueError, KeyError):
            pass

    dump_json(run_dir / "analysis.before.json", warnings)

    return {
        "warnings_before_command": result.model_dump(),
        "warnings_before": warnings,
    }


def _select_warning_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    warnings = _warnings(state.get("warnings_before", []))
    history = _load_history(config)

    # Filter out warnings in ignored paths
    ignore_paths = config.selection_policy.ignore_paths
    ignore_ids = config.selection_policy.ignore_warning_ids
    eligible = [
        w for w in warnings
        if not any(w.file.startswith(p.rstrip("*")) for p in ignore_paths)
        and w.id not in ignore_ids
        and w.severity in config.selection_policy.prefer_severities
    ]

    # Filter out warnings already handled
    if history:
        eligible = [
            w for w in eligible
            if not history.was_completed(w.fingerprint)
            and not history.was_escalated(w.fingerprint)
            and not history.should_skip(w.fingerprint, config.runtime.max_retry_per_warning)
        ]

    if not eligible:
        return {"status": "succeeded", "summary": "No eligible warnings found."}

    # Pick first eligible warning (sorted by severity priority)
    severity_order = {s: i for i, s in enumerate(config.selection_policy.prefer_severities)}
    eligible.sort(key=lambda w: severity_order.get(w.severity, 999))
    selected = eligible[0]

    branch_name = build_branch_name(config, f"{selected.id}-{selected.file.split('/')[-1]}-L{selected.line}")

    # Check escalation rules
    escalation_rules = config.escalation.rules
    should_escalate = False
    escalation_reasons: list[str] = []
    for rule in escalation_rules:
        pattern = rule.get("pattern", "")
        match_type = rule.get("match_type", "")
        if pattern and pattern.strip("*") in (selected.file if match_type == "file" else selected.message):
            should_escalate = True
            escalation_reasons.append(f"Matched escalation rule: {pattern}")

    if should_escalate:
        return {
            "selected_warning": selected.model_dump(),
            "branch_name": branch_name,
            "escalated": True,
            "escalation_reason": "; ".join(escalation_reasons),
        }

    return {
        "selected_warning": selected.model_dump(),
        "branch_name": branch_name,
    }


def _no_warnings_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    run_dir = Path(state["run_dir"])
    dump_text(run_dir / "result.txt", state.get("summary", "No eligible warnings."))

    if config.runtime.audit_log:
        append_audit_entry(config.runtime.audit_log, run_id=state.get("run_id", ""), outcome="no_warnings")

    return {"status": "succeeded"}


def _create_issue_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    warning = _warning(state["selected_warning"])
    title = f"[Auto-fix] {warning.id} in {warning.file}:{warning.line}"
    body = (
        f"## Static analysis warning\n\n"
        f"- **ID**: `{warning.id}`\n"
        f"- **Severity**: `{warning.severity}`\n"
        f"- **File**: `{warning.file}`\n"
        f"- **Line**: {warning.line}\n"
        f"- **Message**: {warning.message}\n"
    )

    if state.get("dry_run"):
        issue = GitHubIssueRef(number=0, url="", title=title)
        return {"issue": issue.model_dump()}

    if not config.github.enabled:
        raise RuntimeError("GitHub is disabled but issue creation is required.")

    client = GitHubClient()
    issue = client.create_issue(
        owner=repo.repo_owner,
        repo=repo.repo_name,
        title=title,
        body=body,
        labels=config.github.labels,
        assignees=config.github.assignees,
    )
    return {"issue": issue.model_dump()}


def _create_worktree_impl(state: AgentState) -> dict[str, Any]:
    repo = _repo(state)
    run_dir = Path(state["run_dir"])
    worktree_path = run_dir / "worktree"
    create_ephemeral_worktree(
        repo_root=Path(repo.root),
        branch_name=state["branch_name"],
        worktree_path=worktree_path,
        start_point=repo.current_branch,
    )
    return {"worktree_path": str(worktree_path)}


def _prepare_llm_task_impl(state: AgentState) -> dict[str, Any]:
    warning = _warning(state["selected_warning"])
    worktree_root = _workspace_root(state)

    # Read the file containing the warning for context
    file_path = worktree_root / warning.file
    file_content = ""
    if file_path.exists():
        file_content = file_path.read_text(encoding="utf-8", errors="replace")

    prompt = (
        f"# Fix static analysis warning\n\n"
        f"## Warning details\n"
        f"- ID: {warning.id}\n"
        f"- Severity: {warning.severity}\n"
        f"- File: {warning.file}\n"
        f"- Line: {warning.line}\n"
        f"- Message: {warning.message}\n\n"
        f"## File content\n```\n{file_content}\n```\n\n"
        f"## Instructions\n"
        f"Fix the warning described above. Make minimal, targeted changes.\n"
        f"Do not introduce new warnings or break existing functionality.\n"
    )

    run_dir = Path(state["run_dir"])
    prompt_path = run_dir / "llm-task.md"
    dump_text(prompt_path, prompt)

    task = LLMTask(
        prompt_path=str(prompt_path),
        prompt_preview=excerpt(prompt, 1200),
        worktree_path=str(worktree_root),
    )
    return {"llm_task": task.model_dump()}


def _run_llm_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    task = LLMTask.model_validate(state["llm_task"])
    worktree_root = Path(task.worktree_path)
    prompt_path = Path(task.prompt_path)

    result = run_junior_task(worktree_root, config.junior_llm, prompt_path)

    run_dir = Path(state["run_dir"])
    dump_json(run_dir / "llm.result.json", result.model_dump())

    if result.returncode != 0:
        return {
            "status": "failed",
            "failure_reason": "LLM CLI failed to execute the fix task.",
            "llm_result": {"command_result": result.model_dump(), "changed_files": []},
        }

    if not has_changes(worktree_root):
        return {
            "status": "failed",
            "failure_reason": "LLM CLI finished without producing any changes.",
            "llm_result": {"command_result": result.model_dump(), "changed_files": []},
        }

    return {"llm_result": {"command_result": result.model_dump(), "changed_files": []}}


def _capture_diff_impl(state: AgentState) -> dict[str, Any]:
    worktree_root = _workspace_root(state)
    changed_files = list_changed_paths(worktree_root)
    if not changed_files:
        return {"status": "failed", "failure_reason": "No modified files found after LLM execution."}

    diff_text, stat_text = capture_diff(worktree_root)
    run_dir = Path(state["run_dir"])
    dump_text(run_dir / "changes.diff", diff_text)
    dump_text(run_dir / "changes.stat", stat_text)

    return {
        "llm_changed_files": changed_files,
        "llm_diff_path": str(run_dir / "changes.diff"),
        "llm_diff_stat_path": str(run_dir / "changes.stat"),
    }


def _build_impl(state: AgentState) -> dict[str, Any]:
    from ..shared.tools.shell import run_command

    config = _cfg(state)
    worktree_root = _workspace_root(state)

    if not config.build.command:
        return {"build_results": []}

    commands = [config.build.command]
    if config.build.setup_command:
        commands.insert(0, config.build.setup_command)

    results = []
    for cmd in commands:
        result = run_command(cmd, cwd=worktree_root, timeout=config.runtime.command_timeout_seconds, extra_env=config.build.env)
        results.append(result.model_dump())
        if result.returncode != 0:
            return {
                "status": "failed",
                "failure_reason": "Build failed.",
                "build_results": results,
            }

    return {"build_results": results}


def _test_impl(state: AgentState) -> dict[str, Any]:
    from ..shared.tools.shell import run_command

    config = _cfg(state)
    worktree_root = _workspace_root(state)

    if not config.tests.commands:
        return {"test_results": []}

    results = []
    for cmd in config.tests.commands:
        result = run_command(cmd, cwd=worktree_root, timeout=config.runtime.command_timeout_seconds, extra_env=config.tests.env)
        results.append(result.model_dump())
        if result.returncode != 0:
            return {
                "status": "failed",
                "failure_reason": "Tests failed.",
                "test_results": results,
            }

    return {"test_results": results}


def _run_analysis_after_impl(state: AgentState) -> dict[str, Any]:
    """Re-run static analysis to verify the warning was fixed and no new ones were introduced."""
    from ..shared.tools.shell import run_command

    config = _cfg(state)
    before = _warnings(state["warnings_before"])
    selected = _warning(state["selected_warning"])
    worktree_root = _workspace_root(state)
    run_dir = Path(state["run_dir"])

    import shlex as _shlex

    if not config.static_analysis.command:
        return {"status": "failed", "failure_reason": "No static_analysis.command configured."}

    source_paths = " ".join(_shlex.quote(p) for p in config.static_analysis.source_paths)
    command = config.static_analysis.command.replace("{source_paths}", source_paths)
    result = run_command(command, cwd=worktree_root, timeout=config.runtime.command_timeout_seconds)

    # Parse warnings using the same format as _run_analysis_before_impl
    after_warnings: list[dict] = []

    if config.static_analysis.output_format == "xml":
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(result.stdout)
            for error in root.iter("error"):
                for location in error.iter("location"):
                    w = WarningCandidate(
                        id=error.get("id", ""),
                        severity=error.get("severity", ""),
                        file=location.get("file", ""),
                        line=int(location.get("line", "0")),
                        message=error.get("msg", ""),
                    )
                    after_warnings.append(w.model_dump())
                    break
        except ET.ParseError:
            pass
    elif config.static_analysis.output_format == "json":
        import json as _json
        try:
            items = _json.loads(result.stdout)
            if isinstance(items, list):
                for item in items:
                    w = WarningCandidate(
                        id=str(item.get("id", item.get("rule", item.get("code", "")))),
                        severity=str(item.get("severity", item.get("type", "warning"))),
                        file=str(item.get("file", item.get("path", ""))),
                        line=int(item.get("line", item.get("lineno", 0))),
                        message=str(item.get("message", item.get("msg", ""))),
                    )
                    after_warnings.append(w.model_dump())
        except (ValueError, KeyError):
            pass

    dump_json(run_dir / "analysis.after.json", after_warnings)

    # Check if the targeted warning was fixed
    after_fingerprints = {w.get("fingerprint", "") for w in after_warnings}
    if selected.fingerprint in after_fingerprints:
        return {
            "status": "failed",
            "failure_reason": f"Target warning {selected.fingerprint} still present after fix.",
            "warnings_after": after_warnings,
        }

    # Check for new warnings not present before
    before_fingerprints = {w.fingerprint for w in before}
    new_warnings = [w for w in after_warnings if w.get("fingerprint", "") not in before_fingerprints]
    if new_warnings:
        return {
            "status": "failed",
            "failure_reason": f"Fix introduced {len(new_warnings)} new warning(s).",
            "warnings_after": after_warnings,
        }

    return {
        "warnings_after_command": result.model_dump(),
        "warnings_after": after_warnings,
        "summary": f"Warning {selected.fingerprint} fixed successfully.",
    }


def _commit_and_push_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    warning = _warning(state["selected_warning"])
    worktree_root = _workspace_root(state)

    if state.get("dry_run"):
        return {"summary": (state.get("summary") or "") + "\nDry-run: commit/push skipped."}

    changed_files = [p for p in state.get("llm_changed_files", []) if p.strip()]
    if not changed_files:
        return {"status": "failed", "failure_reason": "No changed files to commit."}

    stage_paths(worktree_root, changed_files)
    if not has_staged_changes(worktree_root):
        return {"status": "failed", "failure_reason": "No staged changes after filtering."}

    message = f"fix(static-analysis): {warning.id} in {warning.file}:{warning.line}"
    commit_sha = commit_all(worktree_root, message)

    if config.runtime.push_after_commit:
        push_branch(worktree_root, config.git_remote, state["branch_name"])

    return {"commit_sha": commit_sha}


def _close_issue_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    warning = _warning(state["selected_warning"])

    # Record success in history
    history = _load_history(config)
    if history:
        history.record(fingerprint=warning.fingerprint, run_id=state.get("run_id", ""), outcome="fixed")

        # Record collateral fixes
        before = _warnings(state.get("warnings_before", []))
        after = _warnings(state.get("warnings_after", []))
        after_fps = {w.fingerprint for w in after}
        for w in before:
            if w.fingerprint != warning.fingerprint and w.fingerprint not in after_fps:
                history.record(fingerprint=w.fingerprint, run_id=state.get("run_id", ""), outcome="fixed")

    if config.runtime.audit_log:
        append_audit_entry(
            config.runtime.audit_log,
            run_id=state.get("run_id", ""),
            warning_fingerprint=warning.fingerprint,
            outcome="fixed",
            commit_sha=state.get("commit_sha", ""),
        )

    if state.get("dry_run") or not config.github.enabled or not config.runtime.close_issue_after_push:
        return {"status": "succeeded"}

    issue = _issue(state)
    if issue.number > 0:
        client = GitHubClient()
        client.comment_issue(
            repo.repo_owner, repo.repo_name, issue.number,
            f"Fixed in {state.get('commit_sha', 'N/A')}. Warning `{warning.fingerprint}` resolved.",
        )
        client.close_issue(repo.repo_owner, repo.repo_name, issue.number)

    return {"status": "succeeded"}


def _failure_handler_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    reason = state.get("failure_reason", "Unclassified failure.")
    run_dir = Path(state["run_dir"])
    dump_text(run_dir / "failure.txt", reason)

    # Record in history
    selected_raw = state.get("selected_warning")
    if selected_raw:
        warning = _warning(selected_raw)
        history = _load_history(config)
        if history:
            history.record(fingerprint=warning.fingerprint, run_id=state.get("run_id", ""), outcome="failed", failure_reason=reason)
        if config.runtime.audit_log:
            append_audit_entry(config.runtime.audit_log, run_id=state.get("run_id", ""), warning_fingerprint=warning.fingerprint, outcome="failed", failure_reason=reason)

    # Comment on issue if one exists
    if not state.get("dry_run") and config.github.enabled and state.get("issue"):
        issue = _issue(state)
        if issue.number > 0:
            client = GitHubClient()
            client.comment_issue(repo.repo_owner, repo.repo_name, issue.number, f"Auto-fix failed: {reason[:500]}")

    # Cleanup worktree
    worktree_path = state.get("worktree_path")
    if worktree_path and Path(worktree_path).exists():
        try:
            remove_worktree(Path(repo.root), Path(worktree_path))
        except Exception:
            pass

    return {"status": "failed"}


def _escalate_warning_impl(state: AgentState) -> dict[str, Any]:
    config = _cfg(state)
    repo = _repo(state)
    warning = _warning(state["selected_warning"])
    reasons = state.get("escalation_reason", "Matched escalation policy")

    if not state.get("dry_run") and config.github.enabled:
        title = f"[Escalation] {warning.id} in {warning.file}:{warning.line}"
        body = f"## Escalated warning\n\n- **Reason**: {reasons}\n- **Warning**: `{warning.fingerprint}`\n"
        labels = config.github.labels + [config.escalation.escalation_label]
        client = GitHubClient()
        client.create_issue(owner=repo.repo_owner, repo=repo.repo_name, title=title, body=body, labels=labels)

    history = _load_history(config)
    if history:
        history.record(fingerprint=warning.fingerprint, run_id=state.get("run_id", ""), outcome="escalated")

    return {"status": "succeeded"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("prepare", _wrap_node("prepare", _prepare_impl))
    graph.add_node("run_analysis_before", _wrap_node("run_analysis_before", _run_analysis_before_impl))
    graph.add_node("select_warning", _wrap_node("select_warning", _select_warning_impl))
    graph.add_node("no_warnings", _wrap_node("no_warnings", _no_warnings_impl))
    graph.add_node("escalate_warning", _wrap_node("escalate_warning", _escalate_warning_impl))
    graph.add_node("create_issue", _wrap_node("create_issue", _create_issue_impl))
    graph.add_node("create_worktree", _wrap_node("create_worktree", _create_worktree_impl))
    graph.add_node("prepare_llm_task", _wrap_node("prepare_llm_task", _prepare_llm_task_impl))
    graph.add_node("run_llm", _wrap_node("run_llm", _run_llm_impl))
    graph.add_node("capture_diff", _wrap_node("capture_diff", _capture_diff_impl))
    graph.add_node("build_code", _wrap_node("build_code", _build_impl))
    graph.add_node("run_tests", _wrap_node("run_tests", _test_impl))
    graph.add_node("run_analysis_after", _wrap_node("run_analysis_after", _run_analysis_after_impl))
    graph.add_node("commit_and_push", _wrap_node("commit_and_push", _commit_and_push_impl))
    graph.add_node("close_issue", _wrap_node("close_issue", _close_issue_impl))
    graph.add_node("handle_failure", _wrap_node("handle_failure", _failure_handler_impl))

    # --- Edges ---
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", _route_standard, {"ok": "run_analysis_before", "fail": "handle_failure"})
    graph.add_conditional_edges("run_analysis_before", _route_standard, {"ok": "select_warning", "fail": "handle_failure"})
    graph.add_conditional_edges("select_warning", _route_after_select, {
        "ok": "create_issue",
        "done": "no_warnings",
        "escalate": "escalate_warning",
        "fail": "handle_failure",
    })
    graph.add_edge("no_warnings", END)
    graph.add_edge("escalate_warning", END)
    graph.add_conditional_edges("create_issue", _route_standard, {"ok": "create_worktree", "fail": "handle_failure"})
    graph.add_conditional_edges("create_worktree", _route_standard, {"ok": "prepare_llm_task", "fail": "handle_failure"})
    graph.add_conditional_edges("prepare_llm_task", _route_standard, {"ok": "run_llm", "fail": "handle_failure"})
    graph.add_conditional_edges("run_llm", _route_standard, {"ok": "capture_diff", "fail": "handle_failure"})
    graph.add_conditional_edges("capture_diff", _route_standard, {"ok": "build_code", "fail": "handle_failure"})
    graph.add_conditional_edges("build_code", _route_standard, {"ok": "run_tests", "fail": "handle_failure"})
    graph.add_conditional_edges("run_tests", _route_standard, {"ok": "run_analysis_after", "fail": "handle_failure"})
    graph.add_conditional_edges("run_analysis_after", _route_standard, {"ok": "commit_and_push", "fail": "handle_failure"})
    graph.add_conditional_edges("commit_and_push", _route_standard, {"ok": "close_issue", "fail": "handle_failure"})
    graph.add_edge("close_issue", END)
    graph.add_edge("handle_failure", END)

    return graph.compile()
