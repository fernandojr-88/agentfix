"""GitHub REST API client for issue management."""

from __future__ import annotations

import os

import requests

from ..models import GitHubIssueRef


class GitHubClient:
    """Minimal GitHub API client for issue CRUD operations."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN environment variable is not set.")

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(self, method: str, url: str, payload: dict | None = None) -> dict:
        response = requests.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"GitHub API returned {response.status_code}: {response.text}"
            )
        return response.json() if response.text else {}

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> GitHubIssueRef:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        payload = {
            "title": title,
            "body": body,
            "labels": labels or [],
            "assignees": assignees or [],
        }
        data = self._request("POST", url, payload)
        return GitHubIssueRef(number=data["number"], url=data.get("html_url", ""), title=data["title"])

    def comment_issue(self, owner: str, repo: str, issue_number: int, body: str) -> None:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        self._request("POST", url, {"body": body})

    def close_issue(self, owner: str, repo: str, issue_number: int) -> None:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        self._request("PATCH", url, {"state": "closed"})

    def add_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/labels"
        self._request("POST", url, {"labels": [label]})
