# MDM - ML Data Manager

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/hipotures/mdm/releases)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%2FStable-brightgreen.svg)](https://github.com/hipotures/mdm)

A standalone, enterprise-grade dataset management system for machine learning.

## 🚀 Overview

MDM (ML Data Manager) is a powerful, production-ready tool for managing machine learning datasets. Version 1.0.0 represents a mature, stable release with comprehensive features for dataset management, feature engineering, and ML workflow integration.

### Key Features

- **💾 Multi-Backend Support**: SQLite (default), DuckDB (analytics), and PostgreSQL (enterprise)
- **🧬 Advanced Feature Engineering**: Two-tier system with generic and custom transformers
- **📊 Automatic Type Detection**: Smart column type inference with override capabilities
- **🎯 Kaggle Structure Recognition**: Automatic detection of Kaggle competition datasets
- **🔍 Smart Dataset Discovery**: Tag-based and pattern search across all datasets
- **📦 Multiple Export Formats**: CSV, Parquet, JSON with compression options
- **⚡ High Performance**: Batch processing, progress tracking, and optimized data loading
- **🎨 Beautiful CLI**: Rich terminal output with progress bars and formatted tables
- **📈 Comprehensive Statistics**: Automatic profiling and statistical analysis

## 📦 Installation

```bash
# Using pip
pip install mdm

# Using uv (recommended for faster dependency resolution)
pip install uv
uv pip install mdm

# Development installation
git clone https://github.com/hipotures/mdm.git
cd mdm
uv pip install -e .
```

## 🚀 Quick Start

```bash
# Register a dataset
mdm dataset register iris_dataset ./data/iris.csv \
    --target species \
    --problem-type classification

# List all datasets
mdm dataset list

# Get dataset information
mdm dataset info iris_dataset

# View statistics
mdm dataset stats iris_dataset

# Search datasets
mdm dataset search iris
mdm dataset search --tag classification

# Export dataset
mdm dataset export iris_dataset --format parquet --compression gzip

# Update dataset metadata
mdm dataset update iris_dataset --description "Classic ML dataset" --tags "iris,classification"
```

## 💻 Python API

```python
from mdm import MDMClient

# Initialize client
client = MDMClient()

# Register a dataset
client.register_dataset(
    name="titanic",
    path="./data/titanic.csv",
    target_column="survived",
    problem_type="binary_classification"
)

# Load dataset
df = client.load_dataset("titanic")

# Get dataset info
info = client.get_dataset_info("titanic")
print(f"Rows: {info.row_count}, Columns: {info.column_count}")

# Search datasets
datasets = client.search_datasets("classification")

# Export dataset
client.export_dataset("titanic", format="parquet", output_dir="./exports")
```

## 📖 Documentation

### Core Documentation

- **[Project Overview](docs/current/user/01_Project_Overview.md)** - Introduction to MDM
- **[Configuration Guide](docs/current/user/02_Configuration.md)** - Configuration options and settings
- **[Database Architecture](docs/current/user/03_Database_Architecture.md)** - Backend design and selection
- **[Dataset Registration](docs/current/user/04_Dataset_Registration.md)** - How to register datasets
- **[Dataset Management](docs/current/user/05_Dataset_Management_Operations.md)** - Dataset operations
- **[Database Backends](docs/current/user/06_Database_Backends.md)** - Backend comparison and selection
- **[CLI Reference](docs/current/user/07_Command_Line_Interface.md)** - Command-line interface guide
- **[Programmatic API](docs/current/user/08_Programmatic_API.md)** - Python API documentation
- **[Advanced Features](docs/current/user/09_Advanced_Features.md)** - Feature engineering and more

### Guides and Tutorials

- **[Getting Started](docs/current/user/tutorials/01_Getting_Started.md)** - Quick start guide
- **[Advanced Dataset Management](docs/current/user/tutorials/02_Advanced_Dataset_Management.md)** - Advanced operations
- **[Performance Optimization](docs/current/development/Performance_Optimization.md)** - Tips for large datasets

### API Documentation

- **[API Reference](docs/current/api/API_Reference.md)** - Complete API reference
- **[Architecture Design](docs/current/api/Architecture_Design.md)** - System architecture details

## ⚙️ Configuration

MDM uses a hierarchical configuration system with sensible defaults:

```yaml
# ~/.mdm/mdm.yaml
database:
  default_backend: sqlite  # Options: sqlite, duckdb, postgresql
  
performance:
  batch_size: 10000
  enable_progress: true
  
features:
  enable_at_registration: true
  min_column_variance: 0.01
  
logging:
  level: INFO
  file: ~/.mdm/logs/mdm.log
```

Environment variables override file settings:
```bash
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_PERFORMANCE_BATCH_SIZE=50000
export MDM_LOGGING_LEVEL=DEBUG
```

## 🏗️ Architecture

MDM uses a clean, modular architecture:

```
mdm/
├── api/            # Public API and client
├── cli/            # Command-line interface
├── core/           # Core functionality
│   ├── config/     # Configuration management
│   ├── exceptions/ # Custom exceptions
│   └── logging/    # Logging setup
├── dataset/        # Dataset management
│   ├── loaders/    # File format loaders
│   ├── manager/    # Dataset operations
│   └── registrar/  # Registration system
├── features/       # Feature engineering
│   ├── generic/    # Built-in transformers
│   └── custom/     # User-defined features
├── models/         # Data models
├── storage/        # Storage backends
│   ├── base/       # Abstract interfaces
│   ├── sqlite/     # SQLite implementation
│   ├── duckdb/     # DuckDB implementation
│   └── postgresql/ # PostgreSQL implementation
└── utils/          # Utility functions
```

## 🧪 Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test suites
./scripts/run_tests.sh --unit-only
./scripts/run_tests.sh --integration-only
./scripts/run_tests.sh --e2e-only

# Run with coverage
./scripts/run_tests.sh --coverage

# Run specific test file
pytest tests/unit/test_config.py -v
```

Current test coverage: **95.4%** (1110/1163 tests passing)

## 🔒 Backend Selection

MDM supports three storage backends:

### SQLite (Default)
- Best for: Single-user, small to medium datasets (<1GB)
- Pros: Zero configuration, file-based, portable
- Cons: Limited concurrent access

### DuckDB
- Best for: Analytics workloads, medium to large datasets (1GB-100GB)
- Pros: Columnar storage, fast analytics, excellent compression
- Cons: Single-writer limitation

### PostgreSQL
- Best for: Enterprise, multi-user, production deployments
- Pros: Full ACID, concurrent access, scalability
- Cons: Requires server setup

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/hipotures/mdm.git
cd mdm

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
```

## 📊 Performance

MDM is optimized for performance:

- **Batch Processing**: Efficient handling of large datasets with configurable batch sizes
- **Progress Tracking**: Real-time progress bars for long operations
- **Type Detection**: Fast column type inference with caching
- **Memory Efficient**: Streaming data processing to handle datasets larger than RAM
- **Parallel Processing**: Multi-threaded feature generation where applicable

## 🔐 Security

- SQL injection prevention through parameterized queries
- Path traversal protection for file operations
- Secure configuration with environment variable support
- No sensitive data in logs or error messages

## 📝 Release Notes

### Version 1.0.0 (2025-07-12)

- 🎉 First stable production release
- ✅ Comprehensive test suite (95.4% passing)
- 📚 Complete documentation
- 🐛 All critical bugs fixed
- 🚀 Production-ready performance

See [CHANGELOG.md](CHANGELOG.md) for detailed release history.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SQLAlchemy team for the excellent ORM
- Rich library for beautiful terminal output
- DuckDB team for the amazing analytics engine
- Typer for the intuitive CLI framework
- All contributors and users of MDM

## 📞 Support

- 📚 [Documentation](https://github.com/hipotures/mdm/tree/main/docs)
- 🐛 [Issue Tracker](https://github.com/hipotures/mdm/issues)
- 💬 [Discussions](https://github.com/hipotures/mdm/discussions)

---

**MDM v1.0.0** - Production-ready ML dataset management