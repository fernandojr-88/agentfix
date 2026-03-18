"""Shared data models used across all agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """Result of a shell command execution."""

    command: str
    cwd: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0

    def ok(self) -> bool:
        return self.returncode == 0


class GitHubIssueRef(BaseModel):
    """Reference to a GitHub issue."""

    number: int
    url: str = ""
    title: str = ""


class RepoMetadata(BaseModel):
    """Metadata about the git repository being operated on."""

    root: str
    current_branch: str
    remote_url: str
    repo_owner: str
    repo_name: str


class LLMTask(BaseModel):
    """A task to be executed by an LLM CLI tool."""

    prompt_path: str
    prompt_preview: str = ""
    worktree_path: str


class LLMExecution(BaseModel):
    """Result of an LLM CLI execution."""

    command_result: CommandResult
    changed_files: list[str] = Field(default_factory=list)
    diff_path: str = ""
    diff_stat_path: str = ""
