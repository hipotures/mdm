# MDM - ML Data Manager

A standalone, enterprise-grade dataset management system for machine learning.

## Overview

MDM (ML Data Manager) is a powerful tool for managing machine learning datasets with features including:

- **Decentralized Architecture**: Each dataset is self-contained with its own database
- **Multi-Backend Support**: SQLite, DuckDB, and PostgreSQL via SQLAlchemy
- **Automated Feature Engineering**: Two-tier feature generation system
- **Rich Metadata Tracking**: Comprehensive dataset information and quality metrics
- **Export/Import Capabilities**: Support for CSV, Parquet, and JSON formats
- **Command-Line Interface**: Intuitive CLI with rich formatting

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Quick Start

```bash
# Register a dataset
mdm dataset register my_dataset /path/to/data.csv

# List all datasets
mdm dataset list

# Get dataset information
mdm dataset info my_dataset

# Export dataset
mdm dataset export my_dataset --format parquet
```

## Documentation

See the `docs/` directory for comprehensive documentation:

- [Table of Contents](docs/00_Table_of_Contents.md)
- [Project Overview](docs/01_Project_Overview.md)
- [Configuration Guide](docs/02_Configuration.md)
- [CLI Reference](docs/07_Command_Line_Interface.md)

## Configuration

MDM uses a YAML configuration file located at `~/.mdm/mdm.yaml`. See [mdm.yaml.default](docs/mdm.yaml.default) for all available options.

## License

MIT License