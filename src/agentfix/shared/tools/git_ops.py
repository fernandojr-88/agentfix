"""Git operations: worktree management, branching, committing, pushing."""

from __future__ import annotations

import shlex
from pathlib import Path

from .shell import run_command


def ensure_clean_worktree(repo_root: Path) -> None:
    result = run_command("git status --porcelain", cwd=repo_root, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to check worktree status.\n{result.stderr}")
    if result.stdout.strip():
        raise RuntimeError("Repository worktree is not clean.")


def create_ephemeral_worktree(
    repo_root: Path,
    branch_name: str,
    worktree_path: Path,
    start_point: str,
) -> None:
    """Create a temporary git worktree on a new branch."""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    wt_posix = str(worktree_path).replace("\\", "/")
    command = (
        f"git -c core.longpaths=true worktree add --force -B {shlex.quote(branch_name)} "
        f"{shlex.quote(wt_posix)} {shlex.quote(start_point)}"
    )
    result = run_command(command, cwd=repo_root, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create ephemeral worktree.\n{result.stderr}")


def has_changes(repo_root: Path) -> bool:
    result = run_command("git status --porcelain", cwd=repo_root, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to check for changes.\n{result.stderr}")
    return bool(result.stdout.strip())


def list_changed_paths(repo_root: Path) -> list[str]:
    tracked = run_command("git diff --name-only --relative HEAD", cwd=repo_root, timeout=30)
    if tracked.returncode != 0:
        raise RuntimeError(f"Failed to list tracked changes.\n{tracked.stderr}")

    untracked = run_command("git ls-files --others --exclude-standard", cwd=repo_root, timeout=30)
    if untracked.returncode != 0:
        raise RuntimeError(f"Failed to list untracked files.\n{untracked.stderr}")

    paths = {line.strip() for line in tracked.stdout.splitlines() if line.strip()}
    paths.update(line.strip() for line in untracked.stdout.splitlines() if line.strip())
    return sorted(paths)


def capture_diff(repo_root: Path) -> tuple[str, str]:
    diff_result = run_command("git diff --binary", cwd=repo_root, timeout=120)
    if diff_result.returncode != 0:
        raise RuntimeError(f"Failed to capture diff.\n{diff_result.stderr}")

    stat_result = run_command("git diff --stat", cwd=repo_root, timeout=60)
    if stat_result.returncode != 0:
        raise RuntimeError(f"Failed to capture diff stat.\n{stat_result.stderr}")

    return diff_result.stdout, stat_result.stdout


def stage_paths(repo_root: Path, paths: list[str]) -> None:
    if not paths:
        raise RuntimeError("No files provided for staging.")
    quoted = " ".join(shlex.quote(path) for path in paths)
    result = run_command(f"git add -A -- {quoted}", cwd=repo_root, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to stage files.\n{result.stderr}")


def has_staged_changes(repo_root: Path) -> bool:
    result = run_command("git diff --cached --name-only", cwd=repo_root, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to check staged changes.\n{result.stderr}")
    return bool(result.stdout.strip())


def commit_all(repo_root: Path, message: str) -> str:
    commit_result = run_command(
        f"git commit -m {shlex.quote(message)}",
        cwd=repo_root,
        timeout=120,
    )
    if commit_result.returncode != 0:
        raise RuntimeError(f"git commit failed.\n{commit_result.stderr}")

    sha_result = run_command("git rev-parse HEAD", cwd=repo_root, timeout=30)
    if sha_result.returncode != 0:
        raise RuntimeError(f"Failed to retrieve commit SHA.\n{sha_result.stderr}")
    return sha_result.stdout.strip()


def remove_worktree(repo_root: Path, worktree_path: Path) -> None:
    wt_posix = str(worktree_path).replace("\\", "/")
    result = run_command(
        f"git worktree remove --force {shlex.quote(wt_posix)}",
        cwd=repo_root,
        timeout=60,
    )
    if result.returncode != 0:
        run_command("git worktree prune", cwd=repo_root, timeout=30)


def merge_branch_to_current(repo_root: Path, branch_name: str) -> str:
    result = run_command(
        f"git merge --no-edit {shlex.quote(branch_name)}",
        cwd=repo_root,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to merge '{branch_name}'.\n{result.stderr}")

    sha_result = run_command("git rev-parse HEAD", cwd=repo_root, timeout=30)
    return sha_result.stdout.strip()


def delete_local_branch(repo_root: Path, branch_name: str) -> None:
    run_command(f"git branch -d {shlex.quote(branch_name)}", cwd=repo_root, timeout=30)


def delete_remote_branch(repo_root: Path, remote_name: str, branch_name: str) -> None:
    run_command(
        f"git push {shlex.quote(remote_name)} --delete {shlex.quote(branch_name)}",
        cwd=repo_root,
        timeout=120,
    )


def push_branch(repo_root: Path, remote_name: str, branch_name: str) -> None:
    result = run_command(
        f"git push -u {shlex.quote(remote_name)} {shlex.quote(branch_name)}",
        cwd=repo_root,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git push failed.\n{result.stderr}")
