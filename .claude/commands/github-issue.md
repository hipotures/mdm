# GitHub Issue Resolution Workflow

## Objective
Process all open issues in the `hipotures/mdm` repository systematically using MCP GitHub integration.

## Sequential Workflow
Execute the following steps for each issue in order:

### 1. Issue Analysis Phase
- Fetch the next open issue from the repository
- Read the issue description, comments, and any linked discussions thoroughly
- Analyze the problem scope and categorize it as:
  - **Bug fix**: Code defect requiring correction
  - **Feature enhancement**: New functionality addition
  - **Documentation**: Updates to docs, README, etc.
  - **Architecture issue**: Fundamental design problem requiring structural changes

### 2. Problem Assessment
- Identify the root cause of the issue
- Determine required files/modules that need modification
- Assess complexity and estimated effort
- Check for dependencies on other issues or external factors

### 3. Resolution Strategy
**For standard issues (bugs, features, docs):**
- Implement the necessary changes
- Test the solution locally if applicable
- Ensure code follows existing patterns and conventions

**For architecture issues:**
- **STOP and request guidance**
- Provide a detailed analysis including:
  - What architectural limitation is causing the problem
  - What options are available to resolve it
  - Potential impact on existing functionality
  - Recommended approach with pros/cons
- Wait for explicit approval before proceeding

### 4. Implementation Phase
- Make the required code/documentation changes
- Follow best practices for the specific technology stack
- Ensure changes are minimal and focused on the issue at hand
- Add appropriate comments where necessary

### 5. Commit and Documentation
- Create a clear, descriptive commit message following this format:
  ```
  Fix #[issue_number]: [Brief description]
  
  [Detailed explanation of changes made]
  [Any breaking changes or migration notes]
  ```
- Commit the changes to the repository

### 6. Issue Closure
- Close the issue with a comment containing:
  - Reference to the commit ID that resolves it
  - Brief summary of what was fixed/implemented
  - Any additional notes for users (if applicable)

### 7. Continue to Next Issue
- Move to the next open issue and repeat the process
- Maintain a log of processed issues for tracking

## MCP GitHub API Limits

### Issue Fetching Constraints
- **Maximum per_page**: Use `per_page: 5` when fetching issues
- **Token limit**: GitHub API responses are limited to 25,000 tokens maximum
- **Pagination**: Always use small batches and paginate through results
- **Example correct call**: `list_issues(owner: "hipotures", repo: "mdm", state: "open", per_page: 5, page: 1)`

### Recommended Fetching Strategy
```
1. Start with: per_page: 5, page: 1
2. Process those 5 issues completely
3. Then fetch: per_page: 5, page: 2
4. Continue until no more issues
```

**❌ DON'T DO THIS:**
```
list_issues(per_page: 100)  // Will exceed token limit
list_issues(per_page: 20)   // Likely to exceed token limit
```

**✅ DO THIS:**
```
list_issues(per_page: 5, page: 1)  // Safe, won't exceed limits
```

## Guidelines and Best Practices

### Code Quality
- Maintain existing code style and conventions
- Add unit tests for new functionality when appropriate
- Update documentation when making functional changes
- Ensure backwards compatibility unless explicitly stated otherwise

### Communication
- Be clear and concise in commit messages and issue comments
- Reference related issues when applicable using `#issue_number`
- Provide context for future maintainers

### Error Handling
- If an issue cannot be reproduced, ask for clarification in comments
- If an issue requires external dependencies, document the requirements
- If multiple solutions are possible, explain your choice

### Repository-Specific Notes
- Repository: `hipotures/mdm`
- Use MCP GitHub integration for all repository interactions
- Follow the project's existing branching strategy
- Respect any CI/CD pipeline requirements

## Stopping Conditions
- All open issues have been processed
- An architecture issue requires human decision
- An issue cannot be resolved due to missing information or external dependencies

## Success Criteria
- Each issue is either resolved with working code or properly escalated
- All commits are properly documented and linked to their respective issues
- The repository is left in a stable, working state
- No new issues are introduced by the changes made
