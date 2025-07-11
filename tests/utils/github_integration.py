"""Shared GitHub integration utilities for test scripts."""

import os
import hashlib
from pathlib import Path
from typing import Optional, Set, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import time

try:
    from github import Github, GithubException
    from github.Issue import Issue
    from github.Repository import Repository
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None
    GithubException = None
    Issue = None
    Repository = None

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available, will use system environment


@dataclass
class GitHubConfig:
    """GitHub configuration from environment."""
    token: Optional[str]
    repo: str
    rate_limit: int
    
    @classmethod
    def from_env(cls) -> 'GitHubConfig':
        """Load configuration from environment variables."""
        return cls(
            token=os.environ.get('GITHUB_TOKEN'),
            repo=os.environ.get('GITHUB_REPO', 'hipotures/mdm'),
            rate_limit=int(os.environ.get('GITHUB_RATE_LIMIT', '30'))
        )


class RateLimiter:
    """Simple rate limiter for GitHub API calls."""
    
    def __init__(self, max_calls_per_hour: int = 30):
        self.max_calls_per_hour = max_calls_per_hour
        self.calls: List[float] = []
    
    def can_call(self) -> bool:
        """Check if we can make another API call."""
        now = time.time()
        # Remove calls older than 1 hour
        self.calls = [t for t in self.calls if now - t < 3600]
        return len(self.calls) < self.max_calls_per_hour
    
    def record_call(self):
        """Record that an API call was made."""
        self.calls.append(time.time())
    
    def wait_time(self) -> float:
        """Get seconds to wait before next call is allowed."""
        if self.can_call():
            return 0
        
        now = time.time()
        oldest_call = min(self.calls)
        wait_seconds = 3600 - (now - oldest_call) + 1
        return max(0, wait_seconds)


class GitHubIssueManager:
    """Manages GitHub issue creation and updates."""
    
    def __init__(self, config: Optional[GitHubConfig] = None):
        """Initialize with configuration."""
        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub not installed. Install with: uv pip install PyGithub")
        
        self.config = config or GitHubConfig.from_env()
        if not self.config.token:
            raise ValueError("GitHub token not found. Set GITHUB_TOKEN in .env file")
        
        self.github = Github(self.config.token)
        self.repo = self.github.get_repo(self.config.repo)
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self._existing_issues: Optional[List[Issue]] = None
    
    @property
    def existing_issues(self) -> List[Issue]:
        """Get cached list of existing issues."""
        if self._existing_issues is None:
            self._existing_issues = list(self.repo.get_issues(state='open'))
        return self._existing_issues
    
    def find_existing_issue(self, issue_id: str) -> Optional[Issue]:
        """Find existing issue by ID in title or body."""
        for issue in self.existing_issues:
            if f"[{issue_id}]" in issue.title or f"Issue ID: {issue_id}" in issue.body:
                return issue
        return None
    
    def create_issue_id(self, test_name: str, error_type: str, category: str = "") -> str:
        """Create deterministic issue ID from test details."""
        content = f"{category}:{test_name}:{error_type}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def create_or_update_issue(
        self,
        title: str,
        body: str,
        labels: List[str],
        issue_id: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Create new issue or update existing one."""
        result = {
            "action": None,
            "issue_number": None,
            "issue_id": issue_id,
            "message": None,
            "rate_limited": False
        }
        
        # Check rate limit
        if not self.rate_limiter.can_call():
            wait_time = self.rate_limiter.wait_time()
            result["rate_limited"] = True
            result["message"] = f"Rate limit reached. Wait {wait_time:.0f} seconds."
            return result
        
        # Add issue ID to title if provided
        if issue_id:
            title = f"{title} [{issue_id}]"
            body += f"\n\n---\n*Issue ID: {issue_id}*"
        
        # Check for existing issue
        existing_issue = None
        if issue_id:
            existing_issue = self.find_existing_issue(issue_id)
        
        if dry_run:
            if existing_issue:
                result["action"] = "would_update"
                result["issue_number"] = existing_issue.number
                result["message"] = f"Would update issue #{existing_issue.number}"
            else:
                result["action"] = "would_create"
                result["message"] = "Would create new issue"
            return result
        
        try:
            if existing_issue:
                # Add comment to existing issue
                comment = f"### Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{body}"
                existing_issue.create_comment(comment)
                result["action"] = "updated"
                result["issue_number"] = existing_issue.number
                result["message"] = f"Updated issue #{existing_issue.number}"
            else:
                # Create new issue
                issue = self.repo.create_issue(title=title, body=body, labels=labels)
                result["action"] = "created"
                result["issue_number"] = issue.number
                result["message"] = f"Created issue #{issue.number}"
            
            self.rate_limiter.record_call()
            
        except GithubException as e:
            result["action"] = "error"
            result["message"] = f"GitHub API error: {e}"
        
        return result
    
    def format_test_failure_issue(
        self,
        test_name: str,
        error_type: str,
        error_message: str,
        category: str,
        file_path: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format a test failure as issue title and body."""
        # Create issue ID
        issue_id = self.create_issue_id(test_name, error_type, category)
        
        # Create title
        title = f"[Test Failure] {test_name} - {error_type}"
        
        # Create body
        body = f"""## Test Failure: {test_name}

**Category:** {category}
**File:** `{file_path}`
**Error Type:** {error_type}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Error Details
```
{error_message}
```

### How to Reproduce
```bash
pytest {file_path}::{test_name}
```
"""
        
        # Add additional info if provided
        if additional_info:
            if "suggested_fix" in additional_info:
                body += f"\n### Suggested Fix\n{additional_info['suggested_fix']}\n"
            
            if "test_output" in additional_info:
                body += f"\n### Test Output\n```\n{additional_info['test_output']}\n```\n"
        
        # Add footer
        body += "\n---\n*This issue was automatically created by the test failure analyzer.*"
        
        # Create labels
        labels = ["test-failure", "automated"]
        
        # Add category label
        if category:
            category_label = category.lower().replace(" ", "-").replace("/", "-")
            labels.append(f"category-{category_label}")
        
        # Add error type label
        error_label = error_type.lower().replace(" ", "-")
        labels.append(f"error-{error_label}")
        
        return {
            "title": title,
            "body": body,
            "labels": labels,
            "issue_id": issue_id
        }


def check_github_availability() -> bool:
    """Check if GitHub integration is available."""
    if not GITHUB_AVAILABLE:
        return False
    
    config = GitHubConfig.from_env()
    return config.token is not None


def get_suggested_fix(error_type: str) -> str:
    """Get suggested fix based on error type."""
    suggestions = {
        "AssertionError": (
            "- Check if test assertions need updating to match current behavior\n"
            "- Verify expected output format hasn't changed\n"
            "- Ensure test data is correct"
        ),
        "AttributeError": (
            "- Check if mocked objects have all required attributes\n"
            "- Verify API hasn't changed\n"
            "- Ensure proper initialization of objects"
        ),
        "TypeError": (
            "- Check function signatures and parameter types\n"
            "- Verify mock configurations\n"
            "- Ensure correct argument passing"
        ),
        "KeyError": (
            "- Check if required keys exist in dictionaries\n"
            "- Verify data structures haven't changed\n"
            "- Ensure proper data initialization"
        ),
        "ModuleNotFoundError": (
            "- Check import statements\n"
            "- Verify module installation\n"
            "- Ensure correct Python path"
        ),
        "ValueError": (
            "- Check input validation\n"
            "- Verify data format expectations\n"
            "- Ensure proper type conversions"
        ),
        "File/Directory not found": (
            "- Check if test fixtures are properly created\n"
            "- Verify path construction\n"
            "- Ensure proper cleanup/setup"
        ),
        "Output mismatch": (
            "- Check if expected output needs updating\n"
            "- Verify output format changes\n"
            "- Ensure proper string matching"
        ),
        "Command failed": (
            "- Check command syntax\n"
            "- Verify command availability\n"
            "- Ensure proper environment setup"
        )
    }
    
    return suggestions.get(error_type, "- Investigate the specific error and root cause")