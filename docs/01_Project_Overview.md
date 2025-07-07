# ML Data Manager (MDM): Project Overview

## Introduction

ML Data Manager (MDM) is a standalone, enterprise-grade dataset management system designed specifically for machine learning workflows. It provides a unified solution for registering, storing, and managing datasets across multiple storage backends while maintaining rich metadata, quality metrics, and usage analytics.

## Key Benefits

- **Decentralized Architecture**: Each dataset is self-contained with its own database
- **Multi-Backend Support**: Choose the best storage solution for your needs
- **Automated Quality Assurance**: Built-in data profiling and validation
- **Performance Optimized**: Intelligent query optimization
- **Change Tracking**: Monitor when datasets are updated
- **Rich Metadata**: Comprehensive tracking of dataset characteristics and usage

## Language Requirement

**IMPORTANT**: All code, documentation, and communication in MDM must be in English:
- **Code**: All variable names, function names, class names, and code comments must be in English
- **Documentation**: All docstrings, README files, and documentation must be in English
- **User Interface**: All CLI output, error messages, and log messages must be in English
- **Configuration**: All configuration files, schemas, and examples must use English
- **Development**: All git commits, pull requests, and issue descriptions must be in English

This ensures consistency, maintainability, and accessibility for the international development community.

## Use Cases

- **ML Teams**: Manage datasets across multiple projects and experiments
- **Data Scientists**: Quick dataset discovery and quality assessment
- **MLOps**: Dataset tracking and management for production pipelines
- **Research**: Reproducible experiments with dataset snapshots
- **Enterprise**: Distributed governance and monitoring of ML data assets

## System Overview

The ML Data Manager provides a comprehensive system for managing datasets through a simple two-layer architecture:

1. **Dataset-Specific Databases** - Each dataset has its own database file containing both data and metadata
2. **Feature Engineering Cache** - Fast access for ML operations and temporary transformations

Dataset discovery is achieved through a directory-based approach, where each dataset exists as a self-contained unit in `~/.mdm/datasets/`, with YAML configuration files serving as lightweight pointers.

### Key Capabilities

- **Single-Backend Architecture**: SQLite (default), DuckDB, or PostgreSQL via SQLAlchemy. MDM uses one backend type for all datasets at a time.
- **Automatic File Detection**: Scans directories for standard dataset files
- **Automatic Format Detection**: Intelligently detects file formats, encodings, delimiters, and compression
- **Rich Metadata Tracking**: Statistics, quality metrics, usage history
- **Performance Optimization**: Caching, indexing, and query optimization
- **Change Tracking**: Monitor dataset modifications and updates
- **Multi-File Support**: Train, test, validation, submission files

## Architecture Philosophy

MDM follows a fully decentralized architecture where:
- Each dataset is self-contained in its own directory with embedded metadata
- YAML configuration files in `~/.mdm/config/datasets/` serve as discovery pointers
- Dataset listing works by scanning the `~/.mdm/config/datasets/` directory for YAML files
- No central database or registry service required
- Easy to backup, move, or share datasets by copying directories
- Minimal dependencies for maximum portability and simplicity

## Project Structure

MDM follows a standard Python project layout with clear separation of concerns:

```
mdm/
├── src/
│   └── mdm/
│       ├── __init__.py
│       ├── core/           # Core business logic
│       ├── storage/        # Storage backend adapters
│       ├── cli/            # Command-line interface
│       ├── models/         # Data models and schemas
│       ├── utils/          # Utility functions
│       └── config/         # Configuration management
├── tests/                  # Unit and integration tests
│   ├── unit/
│   └── integration/
├── docs/                   # Documentation
│   ├── [0-9][0-9]_*.md    # Main documentation files
│   └── implementation/     # Stage-by-stage implementation guides
├── scripts/                # Helper and E2E test scripts
│   ├── test_e2e_*.sh      # End-to-end test scripts
│   └── utils/             # Development utilities
├── pyproject.toml         # Project configuration
└── uv.lock                # Lock file for reproducible installs (managed by uv)
```

### Key Modules

- **core/**: Contains the main business logic including dataset management, registration, analysis, and batch processing
- **storage/**: Implements the storage abstraction layer with SQLAlchemy-based backend implementations (SQLite, DuckDB, PostgreSQL)
- **cli/**: Provides the command-line interface using Typer and Rich for an enhanced user experience
- **models/**: Defines Pydantic models for data validation and type safety
- **utils/**: Common utilities for logging, formatting, and helper functions
- **config/**: Manages application configuration using Pydantic Settings

## Development Setup

### Virtual Environment and Package Management

MDM uses [`uv`](https://github.com/astral-sh/uv) for BOTH virtual environment creation and package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install MDM and dependencies (MUST use uv pip)
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

**Critical Notes**:
1. **Virtual Environment**: The `.venv` created by `uv venv` is NOT compatible with regular Python/pip
2. **Package Installation**: You MUST use `uv pip`, not regular `pip`
3. **Package Visibility**: Packages installed with regular `pip` will NOT be visible in a uv environment
4. **Lock File**: The `uv.lock` file ensures reproducible installations

**Why uv?**
- Faster dependency resolution and installation (10-100x faster than pip)
- Reproducible builds via `uv.lock`
- Better dependency conflict resolution
- Consistent behavior across platforms
- Integrated virtual environment management

## Getting Started

To begin using MDM:

1. **Configure the system** - See [02_Configuration.md](02_Configuration.md)
2. **Understand the architecture** - See [03_Database_Architecture.md](03_Database_Architecture.md)
3. **Register your first dataset** - See [04_Dataset_Registration.md](04_Dataset_Registration.md)
4. **Explore available operations** - See [05_Dataset_Management_Operations.md](05_Dataset_Management_Operations.md)

## Next Steps

- Continue to [02_Configuration.md](02_Configuration.md) to learn about system configuration
- Jump to [04_Dataset_Registration.md](04_Dataset_Registration.md) for a quick start guide
- See [07_Command_Line_Interface.md](07_Command_Line_Interface.md) for command reference