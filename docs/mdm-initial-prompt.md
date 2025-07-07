# Prompt: Building MDM Application (ML Data Manager)

## Your Task
Build MDM - a dataset management system for machine learning. Follow the specifications in the documentation files and implement the system according to best Python practices.

## CRITICAL: Documentation Sources

1. **Main Documentation**: `docs/[00-13]_*.md` files contain the complete specification
   - Start with `00_Table_of_Contents.md` for navigation
   - Each numbered file covers a specific aspect of the system

2. **Configuration Reference**: `docs/mdm.yaml.default` shows ALL available configuration options
   - Use this as the definitive guide for configuration parameters
   - Every configurable option and its default value is documented here

3. **Implementation Stages**: `docs/implementation/stage_[01-11]_*.md`
   - Build the application stage by stage, NOT all at once
   - Commit after completing each stage

## Build Process

**You MUST build the application in STAGES**:
1. Start with Stage 01 and proceed sequentially through Stage 11
2. After each stage, commit with a descriptive message
3. Test using the Titanic dataset in `tmp/Titanic/` after all stages
4. **CRITICAL**: Complete the manual test checklist (see below)

## Key Requirements

1. **English only** - ALL code, comments, docstrings, and messages must be in English
2. **Decentralized architecture** - No central registry database
3. **Configuration-driven** - Backend selected via `mdm.yaml`, not CLI parameters
4. **Follow the documentation** - Do not deviate from the specified behavior

## Completion Criteria

**The implementation is ONLY complete when:**

1. All 11 implementation stages are finished
2. Tests pass with the Titanic dataset
3. **MOST IMPORTANT**: Every item in `docs/MANUAL_TEST_CHECKLIST.md` is marked as `[X]`

### Verification Process

```bash
# After implementing all stages
cat docs/MANUAL_TEST_CHECKLIST.md

# Execute each test and mark as completed
# Change: - [ ] Test description
# To:     - [X] Test description

# Commit your progress
git add docs/MANUAL_TEST_CHECKLIST.md
git commit -m "✅ Verified [Section Name] tests"
```

**DO NOT CONSIDER THE WORK COMPLETE** until:
- Every single `[ ]` in MANUAL_TEST_CHECKLIST.md is changed to `[X]`
- All 490+ test items are verified and marked as completed
- You have committed the fully completed checklist

## Remember

- Read the documentation thoroughly before implementing
- Use `mdm.yaml.default` as your configuration reference
- Test everything according to MANUAL_TEST_CHECKLIST.md
- The checklist is your final deliverable - 100% completion required

Your success is measured by the completeness of the MANUAL_TEST_CHECKLIST.md file.