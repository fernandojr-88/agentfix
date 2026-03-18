"""CLI for the Compliance Checker agent — single-run and loop modes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ..shared.tools.git_ops import merge_branch_to_current, delete_local_branch, delete_remote_branch
from ..shared.tools.repo import discover_git_root
from ..shared.tools.shell import run_command
from ..shared.utils import dump_json
from .config import load_config
from .graph import build_graph

app = typer.Typer(help="Compliance Checker: plan tasks from requirement docs -> LLM implement -> validate -> commit.")
console = Console()


def _print_summary(final_state: dict) -> None:
    lines = [
        f"status: {final_state.get('status', 'unknown')}",
        f"task_id: {final_state.get('task_id', '-')}",
        f"branch_name: {final_state.get('branch_name', '-')}",
        f"failure_reason: {final_state.get('failure_reason', '-')}",
        f"commit_sha: {final_state.get('commit_sha', '-')}",
        f"is_critical: {final_state.get('is_critical_module', False)}",
        f"tier: {final_state.get('current_tier', '-')}",
    ]
    console.print(Panel("\n".join(lines), title="Run Summary"))


@app.command()
def run(
    config: str = typer.Option("configs/compliance-checker.yaml", help="YAML configuration file."),
    repo: str = typer.Option(".", help="Path to the target repository."),
    dry_run: bool = typer.Option(False, help="Skip issue creation/push."),
    output_json: str = typer.Option("", help="Write final state to this JSON path."),
    loop: bool = typer.Option(False, help="Loop mode: repeat until all tasks are done."),
    max_iterations: int = typer.Option(100, help="Maximum iterations in loop mode."),
    max_consecutive_failures: int = typer.Option(5, help="Stop after N consecutive failures."),
    cooldown_seconds: int = typer.Option(10, help="Pause between loop iterations."),
) -> None:
    if loop:
        _run_loop(
            config_path=config, repo_path=repo, dry_run=dry_run,
            max_iterations=max_iterations, max_consecutive_failures=max_consecutive_failures,
            cooldown_seconds=cooldown_seconds,
        )
        return

    graph = build_graph()
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
    total_completed = 0
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

        _print_summary(final_state)
        status = final_state.get("status", "unknown")
        task = final_state.get("selected_task", {})
        branch = final_state.get("branch_name", "")
        commit_sha = final_state.get("commit_sha", "")
        is_critical = final_state.get("is_critical_module", False)

        result_entry = {"iteration": iteration, "status": status, "task_id": task.get("id", "-")}

        if status == "succeeded" and not task:
            console.print("\n[bold green]All tasks completed![/bold green]")
            results.append(result_entry)
            break

        if status == "succeeded" and final_state.get("escalated"):
            total_escalated += 1
            consecutive_failures = 0
        elif status == "succeeded" and commit_sha:
            total_completed += 1
            consecutive_failures = 0
            # Only merge non-critical branches back to main
            if branch and not is_critical and push_enabled:
                try:
                    merge_branch_to_current(repo_root, branch)
                    run_command(f"git push {remote} main", cwd=repo_root, timeout=600)
                    try:
                        delete_local_branch(repo_root, branch)
                        delete_remote_branch(repo_root, remote, branch)
                    except Exception:
                        pass
                except Exception as exc:
                    console.print(f"  [yellow]Merge skipped:[/yellow] {exc}")
        else:
            total_failed += 1
            consecutive_failures += 1

        results.append(result_entry)
        if consecutive_failures >= max_consecutive_failures:
            break
        time.sleep(cooldown_seconds)

    elapsed = time.time() - start_time
    console.print(f"\n[green]Completed:[/green] {total_completed}  [red]Failed:[/red] {total_failed}  [yellow]Escalated:[/yellow] {total_escalated}  [dim]{elapsed/60:.1f}m[/dim]")

    raise typer.Exit(code=1 if total_failed > 0 and total_completed == 0 else 0)


if __name__ == "__main__":
    app()
