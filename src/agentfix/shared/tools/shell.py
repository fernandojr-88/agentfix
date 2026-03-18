"""Cross-platform shell command execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..models import CommandResult


def _find_bash() -> str:
    """Locate a usable bash executable, handling Windows environments."""
    if sys.platform != "win32":
        return "/bin/bash"
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\msys64\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("bash")
    if found:
        return found
    return "/bin/bash"


_BASH = _find_bash()


def run_command(
    command: str,
    cwd: str | Path,
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    """Execute a shell command via bash and return structured result."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [_BASH, "-c", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
        duration = time.perf_counter() - started
        return CommandResult(
            command=command,
            cwd=str(cwd),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            command=command,
            cwd=str(cwd),
            returncode=124,
            stdout=stdout,
            stderr=stderr + "\nCommand timed out.",
            duration_seconds=duration,
        )
