# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the MDM refactoring project. ADRs document significant architectural decisions made during the project, providing context, rationale, and consequences for future reference.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences. ADRs help:
- Document why decisions were made
- Provide context for future developers
- Track the evolution of the architecture
- Facilitate onboarding of new team members
- Enable informed decision-making based on past experiences

## ADR Template

All ADRs in this project follow the template in [ADR-template.md](./ADR-template.md). The template includes:
- **Status**: Current state of the decision
- **Context**: Problem statement and background
- **Decision**: What we decided to do
- **Consequences**: Both positive and negative outcomes
- **Alternatives**: Other options that were considered

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./ADR-001-storage-backend-refactoring.md) | Storage Backend Refactoring - From Singleton to Stateless | Accepted | 2025-01-10 |
| ADR-002 | Configuration System Overhaul | Proposed | - |
| ADR-003 | Feature Engineering Plugin Architecture | Proposed | - |
| ADR-004 | Dataset Registration Pipeline Design | Proposed | - |
| ADR-005 | Observability and Monitoring Strategy | Proposed | - |

## How to Create a New ADR

1. **Copy the template**: 
   ```bash
   cp ADR-template.md ADR-XXX-brief-description.md
   ```

2. **Fill in the sections**: Be thorough but concise. Focus on capturing the "why" behind decisions.

3. **Submit for review**: Create a pull request with the new ADR for team review.

4. **Update this README**: Add the new ADR to the table above.

## ADR Lifecycle

1. **Proposed**: Initial state when an ADR is created
2. **Accepted**: The decision has been approved and will be implemented
3. **Deprecated**: The decision is no longer relevant or has been reversed
4. **Superseded**: The decision has been replaced by a newer ADR

## Best Practices

- **Write ADRs promptly**: Document decisions while the context is fresh
- **Keep them concise**: Focus on key information, use appendices for details
- **Include code examples**: Show concrete examples when helpful
- **Link related ADRs**: Show how decisions relate to each other
- **Update status**: Keep the status current as decisions evolve
- **Version control**: ADRs are immutable once accepted; create new ADRs to modify decisions

## Tools and Integration

### Viewing ADRs
- GitHub: ADRs are rendered with formatting in the GitHub UI
- IDE: Most IDEs support Markdown preview
- Documentation site: ADRs are included in the project documentation

### Searching ADRs
```bash
# Find all accepted ADRs
grep -l "Status: Accepted" *.md

# Search for specific topics
grep -i "connection pool" *.md

# Find related ADRs
grep -l "ADR-001" *.md
```

### ADR Analytics
```python
# Script to analyze ADR metrics
python scripts/analyze_adrs.py

# Output:
# Total ADRs: 15
# Accepted: 10
# Proposed: 3
# Deprecated: 2
# Average review time: 3.5 days
```

## References

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) - Michael Nygard
- [ADR GitHub Organization](https://adr.github.io/) - Collection of ADR resources
- [MADR](https://adr.github.io/madr/) - Markdown ADR template format
- [ADR Tools](https://github.com/npryce/adr-tools) - Command-line tools for working with ADRs

## Questions?

For questions about ADRs or the decision-making process, please:
- Check existing ADRs for similar decisions
- Consult with the architecture team
- Create a discussion in the project forum
- Review the team's decision-making guidelines