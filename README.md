# agentfix

> Reusable [LangGraph](https://github.com/langchain-ai/langgraph) agents for automated code maintenance.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Three production-ready agents that scan a codebase, delegate fixes to any LLM CLI, validate the results, and commit the changes — all in isolated git worktrees.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Shared Infrastructure                    │
│  models · utils · history · audit · shell · git_ops · llm   │
│  github_api · repo                                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Static        │ │ Test          │ │ Compliance    │
│ Analysis      │ │ Generator     │ │ Checker       │
│ Fixer         │ │               │ │               │
│               │ │ Two-tier      │ │ LLM-planned   │
│ Single-tier   │ │ junior/senior │ │ sub-tasks     │
│ LLM fix       │ │ retry         │ │ + two-tier    │
│               │ │               │ │ + critical    │
│               │ │               │ │   module gate │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Agents

### 1. Static Analysis Fixer

Scans the codebase with any static analysis tool, picks one warning at a time, delegates the fix to an LLM, validates (build + test + re-analysis), and commits.

```
prepare → run_analysis → select_warning → create_issue → create_worktree
→ LLM fix → build → test → verify fix → commit → close_issue
```

- Warning fingerprint tracking (prevents re-attempting fixed/failed warnings)
- Escalation rules (e.g., skip ISR/interrupt code)
- Collateral fix detection (tracks warnings that disappear as side effects)
- Supports any analysis tool via configurable `command` + `output_format` (XML or JSON)

### 2. Test Generator

Scans source modules, generates unit tests via LLM, compiles, runs, measures coverage, and commits. Uses a **two-tier retry** strategy:

1. **Junior tier** (fast LLM — e.g., Codex): Generates the initial test
2. **Senior tier** (powerful LLM — e.g., Claude): Fixes failing tests with expanded context and failure history

```
prepare → scan_modules → select_module → create_issue → create_worktree
→ junior LLM → compile → run → coverage → quality gates → commit
    ↓ (on failure)
  evaluate_tier → senior LLM → re-validate → commit
    ↓ (on failure)
  escalate (create human-review issue)
```

- Module tier classification (pure_logic > mock_able > hw_dependent)
- Coverage delta tracking
- Quality gates (regression tests, optional static analysis)

### 3. Compliance Checker

Reads compliance/regulatory requirement documents, uses an LLM to generate a task plan, then implements each sub-task. Distinguishes **critical modules** (routed to senior tier, branches left open for human review).

```
prepare → load/generate task plan → select_next_task → create_issue
→ create_worktree → junior|senior LLM → validate → commit
→ close_issue (normal) | label_for_review (critical)
    ↓ (on failure)
  evaluate_tier → retry or escalate
```

- LLM-generated task plan from requirement docs
- Source doc hash change detection (auto-regenerate plan when docs change)
- Critical module detection → senior-only + human review gate
- Task plan persistence (YAML)

## Shared Patterns

| Pattern | Description |
|---------|-------------|
| **Ephemeral worktrees** | Each fix runs in an isolated `git worktree` — main branch stays clean |
| **State snapshots** | `state.json` persisted after every node — enables resume on crash |
| **Resume support** | `--resume <run_dir>` skips already-completed nodes |
| **Loop mode** | `--loop` runs continuously until no work remains or max failures hit |
| **History tracking** | JSON file tracks per-item outcomes (fixed/failed/escalated) |
| **Audit log** | JSONL append-only log for compliance/debugging |
| **GitHub integration** | Auto-create issues, comment progress, close on success |
| **Escalation** | After N failures → create "needs-human-review" issue and move on |
| **Dry-run** | `--dry-run` skips GitHub API calls and git push |

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Static analysis fixer (dry-run)
agentfix-static-analysis --config configs/static-analysis.yaml --dry-run

# Test generator (dry-run)
agentfix-test-generator --config configs/test-generator.yaml --dry-run

# Compliance checker (dry-run)
agentfix-compliance --config configs/compliance-checker.yaml --dry-run
```

### Loop mode

```bash
agentfix-static-analysis --config configs/static-analysis.yaml \
  --loop --max-iterations 50 --cooldown-seconds 10
```

### Resume after crash

```bash
agentfix-static-analysis --config configs/static-analysis.yaml \
  --resume ./artifacts/static-analysis/20260317T120000Z
```

## Configuration

Each agent reads a YAML config file. See [`configs/`](configs/) for examples.

### Environment variables

Create a `.env` file (see [`.env.example`](.env.example)):

```
GITHUB_TOKEN=ghp_...
ANTHROPIC_API_KEY=sk-ant-...
```

Config values support `${VAR}` expansion from environment.

### LLM CLI backends

The agents are **CLI-agnostic**. The `junior_llm.binary` and `senior_llm.binary` settings point to any CLI tool that:

1. Reads a prompt from stdin or a file
2. Makes code changes in the working directory
3. Returns exit code 0 on success

Tested with: [Codex CLI](https://github.com/openai/codex), [Claude Code](https://claude.ai/claude-code), [aider](https://aider.chat).

## Extending

### Custom static analysis tool

Set `static_analysis.command` in your YAML config. Use `{source_paths}` as a placeholder:

```yaml
static_analysis:
  command: "pylint --output-format=json {source_paths}"
  output_format: json
  source_paths: [src/]
```

Supported output formats: `xml` (cppcheck-compatible), `json` (pylint/ESLint-compatible).

### Custom test framework

Override `_compile_test_impl`, `_run_test_impl`, and `_measure_coverage_impl` in `test_generator/graph.py`.

### Custom compliance docs

Customize `_generate_task_plan_impl` in `compliance_checker/graph.py` to parse your requirement document format.

## Project Structure

```
agentfix/
├── pyproject.toml
├── .env.example
├── configs/
│   ├── static-analysis.yaml
│   ├── test-generator.yaml
│   └── compliance-checker.yaml
└── src/agentfix/
    ├── shared/
    │   ├── models.py          # CommandResult, GitHubIssueRef, RepoMetadata
    │   ├── config.py          # Shared Pydantic config models
    │   ├── utils.py           # dump_json, excerpt, slugify, etc.
    │   ├── history.py         # ItemHistory (JSON-backed retry tracking)
    │   ├── audit.py           # JSONL audit log
    │   └── tools/
    │       ├── shell.py       # Cross-platform command execution
    │       ├── git_ops.py     # Worktree, branch, commit, push
    │       ├── github_api.py  # GitHub REST API client
    │       ├── llm_cli.py     # Junior/senior LLM CLI wrappers
    │       └── repo.py        # Git repo metadata discovery
    ├── static_analysis/
    │   ├── config.py          # Agent-specific config
    │   ├── state.py           # AgentState TypedDict
    │   ├── models.py          # WarningCandidate
    │   ├── graph.py           # LangGraph StateGraph
    │   └── cli.py             # Typer CLI (single-run + loop)
    ├── test_generator/
    │   ├── config.py
    │   ├── state.py
    │   ├── graph.py           # Two-tier retry graph
    │   └── cli.py
    └── compliance_checker/
        ├── config.py
        ├── state.py
        ├── graph.py           # Task planning + critical module routing
        └── cli.py
```

## License

MIT
