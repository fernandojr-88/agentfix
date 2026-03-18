"""LLM CLI wrappers for junior (Codex) and senior (Claude) tier execution."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import CommandResult
from .shell import run_command

if TYPE_CHECKING:
    from ..config import JuniorLLMSettings, SeniorLLMSettings


_APPROVAL_FLAGS = {
    "full-auto": "--full-auto",
    "auto-edit": "--auto-edit",
    "suggest": "--suggest",
}


def ensure_binary(binary: str) -> None:
    """Verify that an LLM CLI binary is available on PATH."""
    result = run_command(f"command -v {shlex.quote(binary)}", cwd=Path.cwd(), timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"LLM CLI binary not found: {binary}. Ensure it is installed and on PATH.")


def build_junior_command(settings: "JuniorLLMSettings", prompt_path: Path) -> str:
    """Build the command string for the junior-tier LLM CLI (e.g. Codex)."""
    parts = [shlex.quote(settings.binary), "exec"]
    approval_flag = _APPROVAL_FLAGS.get(settings.approval_mode.strip().lower())
    if approval_flag:
        parts.append(approval_flag)
    if settings.model.strip():
        parts.extend(["-m", shlex.quote(settings.model.strip())])
    parts.extend(shlex.quote(arg) for arg in settings.extra_args)
    return f"cat {shlex.quote(str(prompt_path))} | {' '.join(parts)}"


def run_junior_task(
    worktree_root: Path,
    settings: "JuniorLLMSettings",
    prompt_path: Path,
) -> CommandResult:
    """Execute the junior-tier LLM CLI in the given worktree."""
    command = build_junior_command(settings, prompt_path)
    return run_command(
        command,
        cwd=worktree_root,
        timeout=settings.task_timeout_seconds,
        extra_env=settings.env,
    )


def build_senior_command(settings: "SeniorLLMSettings", prompt_path: Path) -> str:
    """Build the command string for the senior-tier LLM CLI (e.g. Claude Code)."""
    parts = [shlex.quote(settings.binary)]
    parts.extend(["-p", shlex.quote(str(prompt_path))])
    parts.append("--output-format=text")
    return " ".join(parts)


def run_senior_task(
    worktree_root: Path,
    settings: "SeniorLLMSettings",
    prompt_path: Path,
) -> CommandResult:
    """Execute the senior-tier LLM CLI in the given worktree."""
    command = build_senior_command(settings, prompt_path)
    return run_command(
        command,
        cwd=worktree_root,
        timeout=settings.task_timeout_seconds,
        extra_env=settings.env,
    )
