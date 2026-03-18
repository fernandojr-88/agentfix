"""CLI for the Static Analysis Fixer agent — single-run and loop modes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..shared.tools.git_ops import (
    delete_local_branch,
    delete_remote_branch,
    merge_branch_to_current,
    remove_worktree,
)
from ..shared.tools.repo import discover_git_root
from ..shared.tools.shell import run_command
from ..shared.utils import dump_json
from .config import load_config
from .graph import build_graph

app = typer.Typer(help="Static Analysis Fixer: scan -> LLM fix -> build/test -> verify -> commit/push.")
console = Console()


def _print_summary(final_state: dict) -> None:
    lines = [
        f"status: {final_state.get('status', 'unknown')}",
        f"run_dir: {final_state.get('run_dir', '-')}",
        f"branch_name: {final_state.get('branch_name', '-')}",
        f"failure_reason: {final_state.get('failure_reason', '-')}",
        f"commit_sha: {final_state.get('commit_sha', '-')}",
    ]
    selected = final_state.get("selected_warning")
    if selected:
        lines.append(f"selected_warning: {selected.get('fingerprint')}")
    console.print(Panel("\n".join(lines), title="Run Summary"))


def _load_resume_state(run_dir: str) -> dict:
    state_path = Path(run_dir) / "state.json"
    if not state_path.exists():
        raise typer.BadParameter(f"state.json not found in: {run_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    last_node = state.get("last_completed_node", "")
    if not last_node:
        raise typer.BadParameter("No last_completed_node in state — cannot determine resume point.")
    if state.get("status") == "failed":
        state["status"] = "running"
        state.pop("failure_reason", None)
        state.pop("failure_traceback", None)
    state["resumed_from"] = last_node
    console.print(f"[bold]Resuming from node:[/bold] {last_node}")
    return state


def _cleanup_stale_worktrees(repo_root: Path) -> None:
    run_command("git worktree prune", cwd=repo_root, timeout=30)
    result = run_command("git worktree list --porcelain", cwd=repo_root, timeout=30)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("worktree ") and "/artifacts/" in line:
                wt_path = line.split("worktree ", 1)[1].strip()
                try:
                    remove_worktree(repo_root, Path(wt_path))
                except Exception:
                    pass
    run_command("git worktree prune", cwd=repo_root, timeout=30)


def _merge_fix_to_main(repo_root: Path, branch_name: str, remote: str, push: bool) -> str | None:
    try:
        sha = merge_branch_to_current(repo_root, branch_name)
        console.print(f"  [green]Merged[/green] {branch_name} -> main ({sha[:8]})")
        if push:
            result = run_command(f"git push {remote} main", cwd=repo_root, timeout=600)
            if result.returncode == 0:
                console.print(f"  [green]Pushed[/green] main to {remote}")
        try:
            delete_local_branch(repo_root, branch_name)
        except Exception:
            pass
        if push:
            try:
                delete_remote_branch(repo_root, remote, branch_name)
            except Exception:
                pass
        return sha
    except Exception as exc:
        console.print(f"  [red]Merge failed:[/red] {exc}")
        return None


@app.command()
def run(
    config: str = typer.Option("configs/static-analysis.yaml", help="YAML configuration file."),
    repo: str = typer.Option(".", help="Path to the target repository."),
    dry_run: bool = typer.Option(False, help="Skip issue creation/push."),
    output_json: str = typer.Option("", help="Write final state to this JSON path."),
    resume: str = typer.Option("", help="Path to a previous run_dir to resume from."),
    loop: bool = typer.Option(False, help="Loop mode: repeat until no warnings remain."),
    max_iterations: int = typer.Option(200, help="Maximum iterations in loop mode."),
    max_consecutive_failures: int = typer.Option(5, help="Stop after N consecutive failures."),
    cooldown_seconds: int = typer.Option(10, help="Pause between loop iterations (seconds)."),
) -> None:
    if loop:
        _run_loop(
            config_path=config, repo_path=repo, dry_run=dry_run,
            max_iterations=max_iterations, max_consecutive_failures=max_consecutive_failures,
            cooldown_seconds=cooldown_seconds,
        )
        return

    graph = build_graph()
    initial_state = _load_resume_state(resume) if resume else {
        "config_path": config, "repo_path": repo, "dry_run": dry_run,
    }
    final_state = graph.invoke(initial_state)
    _print_summary(final_state)
    if output_json:
        dump_json(Path(output_json), final_state)
    raise typer.Exit(code=0 if final_state.get("status") == "succeeded" else 1)


def _run_loop(
    *, config_path: str, repo_path: str, dry_run: bool,
    max_iterations: int, max_consecutive_failures: int, cooldown_seconds: int,
) -> None:
    cfg = load_config(config_path)
    repo_root = discover_git_root(repo_path or cfg.repo_path)
    remote = cfg.git_remote
    push_enabled = cfg.runtime.push_after_commit and not dry_run

    results: list[dict] = []
    consecutive_failures = 0
    total_fixed = 0
    total_failed = 0
    total_escalated = 0
    start_time = time.time()

    console.print(Panel(
        f"Max iterations: {max_iterations}\nMax consecutive failures: {max_consecutive_failures}\n"
        f"Cooldown: {cooldown_seconds}s\nPush enabled: {push_enabled}\nDry run: {dry_run}",
        title="[bold]Loop Mode Activated[/bold]",
    ))

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n{'='*60}\n[bold cyan]Iteration {iteration}/{max_iterations}[/bold cyan]\n{'='*60}")

        graph = build_graph()
        initial_state = {"config_path": config_path, "repo_path": repo_path, "dry_run": dry_run}

        try:
            final_state = graph.invoke(initial_state)
        except Exception as exc:
            console.print(f"[red]Graph crashed:[/red] {exc}")
            _cleanup_stale_worktrees(repo_root)
            consecutive_failures += 1
            total_failed += 1
            results.append({"iteration": iteration, "status": "crashed", "error": str(exc)})
            if consecutive_failures >= max_consecutive_failures:
                break
            time.sleep(cooldown_seconds)
            continue

        status = final_state.get("status", "unknown")
        branch_name = final_state.get("branch_name", "")
        selected = final_state.get("selected_warning", {})
        commit_sha = final_state.get("commit_sha", "")

        _print_summary(final_state)
        result_entry = {
            "iteration": iteration, "status": status,
            "warning_id": selected.get("id", "-"), "warning_file": selected.get("file", "-"),
            "branch_name": branch_name, "commit_sha": commit_sha,
        }

        if status == "succeeded" and not selected:
            console.print("\n[bold green]All eligible warnings processed![/bold green]")
            results.append(result_entry)
            break

        if status == "succeeded" and final_state.get("escalated"):
            total_escalated += 1
            consecutive_failures = 0
        elif status == "succeeded" and commit_sha:
            total_fixed += 1
            consecutive_failures = 0
            if branch_name:
                _merge_fix_to_main(repo_root, branch_name, remote, push_enabled)
        else:
            total_failed += 1
            consecutive_failures += 1
            _cleanup_stale_worktrees(repo_root)

        results.append(result_entry)
        if consecutive_failures >= max_consecutive_failures:
            console.print(f"\n[red bold]Stopping: {max_consecutive_failures} consecutive failures.[/red bold]")
            break
        time.sleep(cooldown_seconds)

    elapsed = time.time() - start_time
    console.print(f"\n{'='*60}\n[bold]Loop Summary[/bold]\n{'='*60}")
    console.print(f"[green]Fixed:[/green] {total_fixed}  [red]Failed:[/red] {total_failed}  [yellow]Escalated:[/yellow] {total_escalated}  [dim]Time: {elapsed/60:.1f}m[/dim]")

    if cfg.runtime.artifacts_dir:
        dump_json(Path(cfg.runtime.artifacts_dir) / "loop_results.json", {
            "total_iterations": len(results), "fixed": total_fixed, "failed": total_failed,
            "escalated": total_escalated, "elapsed_seconds": round(elapsed, 1), "runs": results,
        })

    raise typer.Exit(code=1 if total_failed > 0 and total_fixed == 0 else 0)


if __name__ == "__main__":
    app()
