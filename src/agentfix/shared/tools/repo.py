"""Git repository metadata discovery."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import RepoMetadata
from ..utils import slugify
from .shell import run_command


def discover_git_root(path: str | Path = ".") -> Path:
    result = run_command("git rev-parse --show-toplevel", cwd=Path(path).resolve(), timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"Not a git repository: {path}\n{result.stderr}")
    return Path(result.stdout.strip())


def collect_repo_metadata(repo_root: Path, config: object) -> RepoMetadata:
    """Collect metadata about the repository. Config is expected to have github.repo_owner and repo_name."""
    branch_result = run_command("git branch --show-current", cwd=repo_root, timeout=15)
    current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "main"

    remote_result = run_command("git remote get-url origin", cwd=repo_root, timeout=15)
    remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else ""

    # Extract owner/repo from remote URL or config
    repo_owner = getattr(getattr(config, "github", None), "repo_owner", "") or ""
    repo_name = getattr(getattr(config, "github", None), "repo_name", "") or ""

    if not repo_owner or not repo_name:
        match = re.search(r"[:/]([^/]+)/([^/.]+?)(?:\.git)?$", remote_url)
        if match:
            repo_owner = repo_owner or match.group(1)
            repo_name = repo_name or match.group(2)

    return RepoMetadata(
        root=str(repo_root),
        current_branch=current_branch,
        remote_url=remote_url,
        repo_owner=repo_owner,
        repo_name=repo_name,
    )


def build_branch_name(config: object, identifier: str) -> str:
    """Build a branch name from config prefix and a slugified identifier."""
    prefix = getattr(config, "branch_prefix", "fix")
    slug = slugify(identifier)
    return f"{prefix}/{slug}"
