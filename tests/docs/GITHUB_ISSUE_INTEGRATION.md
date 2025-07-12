# GitHub Issue Integration for Test Failures

## Overview

The `analyze_current_failures.py` script now includes GitHub integration to automatically create and manage issues for test failures.

## Features

- **Automatic Issue Creation**: Creates GitHub issues for each unique error pattern
- **Deduplication**: Prevents duplicate issues by tracking unique failure patterns
- **Issue Updates**: Updates existing issues with new test run results
- **Smart Grouping**: Groups failures by category and error type
- **Priority Labels**: Automatically assigns priority based on failure count
- **Dry Run Mode**: Preview issues before creation

## Setup

### 1. Install Dependencies

```bash
uv pip install -e ".[dev]"
```

This installs PyGithub which is required for GitHub integration.

### 2. Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token" (classic)
3. Give it a descriptive name like "MDM Test Issue Creator"
4. Select scopes:
   - `repo` (full control of private repositories)
   - Or just `public_repo` if MDM is public
5. Click "Generate token" and copy it

### 3. Set Environment Variable

```bash
export GITHUB_TOKEN="your_token_here"
```

Or add to your shell configuration file.

## Usage

### Basic Analysis (No GitHub Integration)

```bash
# Just analyze failures and generate report
python tests/e2e/analyze_current_failures.py
```

### Create GitHub Issues

```bash
# Create issues for all failures
python tests/e2e/analyze_current_failures.py --create-issues

# Dry run - see what would be created
python tests/e2e/analyze_current_failures.py --create-issues --dry-run

# Limit number of issues
python tests/e2e/analyze_current_failures.py --create-issues --max-issues 5

# Use specific token
python tests/e2e/analyze_current_failures.py --create-issues --github-token YOUR_TOKEN
```

### Custom Repository

```bash
# For a fork or different repo
python tests/e2e/analyze_current_failures.py --create-issues --owner myusername --repo myrepo
```

## Issue Format

### Title
```
[E2E Test] {Category}: {Error Type} - {Count} failures
```

Example: `[E2E Test] Dataset Registration: Command failed - 5 failures`

### Labels
Each issue is tagged with:
- `test-failure` - Always added
- `e2e-test` - Always added
- `priority-high/medium/low` - Based on failure count
- `category-{name}` - Test category (e.g., `category-configuration`)
- `error-{type}` - Error type (e.g., `error-command-failed`)

### Body
Issues include:
- Summary with error type and failure count
- List of failed tests with error messages
- Common pattern analysis
- Suggested fixes
- Unique Issue ID for deduplication

## Deduplication

Issues are deduplicated using a hash of:
- Category
- Error type
- Test names (sorted)

This ensures that the same set of failing tests won't create duplicate issues.

## Updates

When tests are run again:
- Existing open issues are found by their Issue ID
- A comment is added with the latest test run results
- No new issue is created if the pattern matches

## Priority Rules

- **High**: 5+ failures
- **Medium**: 2-4 failures  
- **Low**: 1 failure

## Safety Features

1. **Maximum Issues**: Default limit of 10 issues per run
2. **Dry Run**: Preview before creating
3. **Token Validation**: Checks repository access before proceeding
4. **Error Handling**: Graceful failure if GitHub is unavailable

## Example Workflow

```bash
# 1. First run - analyze failures
python tests/e2e/analyze_current_failures.py

# 2. Review the report
cat tests/e2e/CURRENT_TEST_FAILURES.md

# 3. Dry run to see what issues would be created
python tests/e2e/analyze_current_failures.py --create-issues --dry-run

# 4. Create actual issues
python tests/e2e/analyze_current_failures.py --create-issues

# 5. After fixing some tests, run again
# This will update existing issues or create new ones
python tests/e2e/analyze_current_failures.py --create-issues
```

## Troubleshooting

### "PyGithub not installed"
```bash
uv pip install PyGithub
```

### "GitHub token required"
Set the GITHUB_TOKEN environment variable or use --github-token flag

### "Error accessing repository"
- Check token has correct permissions
- Verify owner/repo names are correct
- Ensure token is not expired

### Rate Limiting
GitHub has rate limits. The script respects these and will stop if limits are reached.