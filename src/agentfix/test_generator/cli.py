"""CLI for the Test Generator agent — single-run and loop modes."""

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

app = typer.Typer(help="Test Generator: scan modules -> LLM generate tests -> compile/run/coverage -> commit/push.")
console = Console()


def _print_summary(final_state: dict) -> None:
    lines = [
        f"status: {final_state.get('status', 'unknown')}",
        f"run_dir: {final_state.get('run_dir', '-')}",
        f"branch_name: {final_state.get('branch_name', '-')}",
        f"failure_reason: {final_state.get('failure_reason', '-')}",
        f"commit_sha: {final_state.get('commit_sha', '-')}",
        f"coverage_delta: {final_state.get('coverage_delta', '-')}",
    ]
    selected = final_state.get("selected_module")
    if selected:
        lines.append(f"selected_module: {selected.get('name')} ({selected.get('tier')})")
    console.print(Panel("\n".join(lines), title="Run Summary"))


def _cleanup_stale_worktrees(repo_root: Path) -> None:
    run_command("git worktree prune", cwd=repo_root, timeout=30)


def _merge_to_main(repo_root: Path, branch_name: str, remote: str, push: bool) -> str | None:
    try:
        sha = merge_branch_to_current(repo_root, branch_name)
        console.print(f"  [green]Merged[/green] {branch_name} -> main ({sha[:8]})")
        if push:
            run_command(f"git push {remote} main", cwd=repo_root, timeout=600)
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
    config: str = typer.Option("configs/test-generator.yaml", help="YAML configuration file."),
    repo: str = typer.Option(".", help="Path to the target repository."),
    dry_run: bool = typer.Option(False, help="Skip issue creation/push."),
    output_json: str = typer.Option("", help="Write final state to this JSON path."),
    resume: str = typer.Option("", help="Path to a previous run_dir to resume from."),
    loop: bool = typer.Option(False, help="Loop mode: repeat until all modules are tested."),
    max_iterations: int = typer.Option(100, help="Maximum iterations in loop mode."),
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
    if resume:
        state_path = Path(resume) / "state.json"
        initial_state = json.loads(state_path.read_text(encoding="utf-8"))
        initial_state["resumed_from"] = initial_state.get("last_completed_node", "")
        if initial_state.get("status") == "failed":
            initial_state["status"] = "running"
    else:
        initial_state = {"config_path": config, "repo_path": repo, "dry_run": dry_run}

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
    total_tested = 0
    total_failed = 0
    total_escalated = 0
    start_time = time.time()

    console.print(Panel(
        f"Max iterations: {max_iterations}\nCooldown: {cooldown_seconds}s\n"
        f"Push enabled: {push_enabled}\nDry run: {dry_run}",
        title="[bold]Loop Mode[/bold]",
    ))

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n{'='*60}\n[bold cyan]Iteration {iteration}/{max_iterations}[/bold cyan]\n{'='*60}")

        run_command("git checkout -- .", cwd=repo_root, timeout=30)
        _cleanup_stale_worktrees(repo_root)

        graph = build_graph()
        try:
            final_state = graph.invoke({"config_path": config_path, "repo_path": repo_path, "dry_run": dry_run})
        except Exception as exc:
            console.print(f"[red]Crashed:[/red] {exc}")
            consecutive_failures += 1
            total_failed += 1
            results.append({"iteration": iteration, "status": "crashed"})
            if consecutive_failures >= max_consecutive_failures:
                break
            time.sleep(cooldown_seconds)
            continue

        status = final_state.get("status", "unknown")
        selected = final_state.get("selected_module", {})
        commit_sha = final_state.get("commit_sha", "")
        branch_name = final_state.get("branch_name", "")

        _print_summary(final_state)
        result_entry = {"iteration": iteration, "status": status, "module": selected.get("name", "-")}

        if status == "succeeded" and not selected:
            console.print("\n[bold green]All modules processed![/bold green]")
            results.append(result_entry)
            break

        if status == "succeeded" and commit_sha:
            total_tested += 1
            consecutive_failures = 0
            if branch_name:
                _merge_to_main(repo_root, branch_name, remote, push_enabled)
        elif status == "succeeded" and final_state.get("escalated"):
            total_escalated += 1
            consecutive_failures = 0
        else:
            total_failed += 1
            consecutive_failures += 1

        results.append(result_entry)
        if consecutive_failures >= max_consecutive_failures:
            break
        time.sleep(cooldown_seconds)

    elapsed = time.time() - start_time
    console.print(f"\n[green]Tested:[/green] {total_tested}  [red]Failed:[/red] {total_failed}  [yellow]Escalated:[/yellow] {total_escalated}  [dim]{elapsed/60:.1f}m[/dim]")

    raise typer.Exit(code=1 if total_failed > 0 and total_tested == 0 else 0)


if __name__ == "__main__":
    app()
