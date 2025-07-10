# MDM - ML Data Manager (2025 Refactored Edition)

A standalone, enterprise-grade dataset management system for machine learning with a modern, refactored architecture.

## 🚀 Overview

MDM (ML Data Manager) is a powerful tool for managing machine learning datasets. This 2025 refactored version introduces a clean architecture with interfaces, adapters, and performance optimizations while maintaining backward compatibility.

### Key Features

- **🏗️ Interface-Based Architecture**: Clean separation of concerns with well-defined interfaces
- **🔄 Seamless Migration**: Feature flag-based migration from legacy to new implementation
- **💾 Multi-Backend Support**: SQLite, DuckDB, and PostgreSQL with optimized configurations
- **⚡ Performance Optimizations**: Query optimization, caching, batch processing, and connection pooling
- **🧬 Advanced Feature Engineering**: Two-tier system with custom transformer support
- **📊 Rich Metadata Tracking**: Comprehensive dataset information and quality metrics
- **🔍 Smart Search**: Dataset discovery with tag-based and content search
- **📦 Multiple Export Formats**: CSV, Parquet, JSON with compression options
- **🎨 Beautiful CLI**: Rich terminal output with progress bars and formatted tables

## 📦 Installation

```bash
# Using uv (recommended - faster dependency resolution)
pip install uv
uv pip install mdm-refactor

# Using pip
pip install mdm-refactor

# Development installation
git clone https://github.com/mdm/mdm-refactor-2025.git
cd mdm-refactor-2025
uv pip install -e .
```

## 🚀 Quick Start

```bash
# Initialize MDM
mdm init

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
mdm dataset list --tag classification

# Export dataset
mdm dataset export iris_dataset --format parquet --compression gzip
```

## 📖 Documentation

### Core Documentation

- **[API Reference](docs/API_Reference.md)** - Complete API documentation
- **[Migration Guide](docs/Migration_Guide.md)** - Migrate from legacy MDM
- **[Architecture Decisions](docs/Architecture_Decisions.md)** - Design rationale and patterns
- **[Troubleshooting Guide](docs/Troubleshooting_Guide.md)** - Common issues and solutions

### Tutorials

1. **[Getting Started](docs/tutorials/01_Getting_Started.md)** - Basic MDM usage
2. **[Advanced Dataset Management](docs/tutorials/02_Advanced_Dataset_Management.md)** - Multi-file datasets, time series, large data
3. **[Custom Feature Engineering](docs/tutorials/03_Custom_Feature_Engineering.md)** - Create custom transformers
4. **[Performance Optimization](docs/tutorials/04_Performance_Optimization.md)** - Optimize for large datasets

### Technical Documentation

- [Storage Backend Design](docs/03_Database_Architecture.md)
- [Configuration System](docs/02_Configuration.md)
- [CLI Reference](docs/07_Command_Line_Interface.md)
- [Testing Strategy](docs/test_progress.md)

## ⚙️ Configuration

MDM uses a hierarchical configuration system:

```yaml
# ~/.mdm/mdm.yaml
database:
  default_backend: sqlite  # or duckdb, postgresql
  sqlite:
    pragmas:
      journal_mode: WAL
      synchronous: NORMAL
performance:
  batch_size: 10000
  max_workers: 4
  cache_size_mb: 100
```

Environment variables override file settings:
```bash
export MDM_DATABASE_DEFAULT_BACKEND=duckdb
export MDM_PERFORMANCE_BATCH_SIZE=50000
```

## 🔄 Migration from Legacy MDM

The refactored version maintains full backward compatibility while offering new features:

```python
# Enable new features gradually
from mdm.core import feature_flags

# Start with legacy implementation
feature_flags.set("use_new_storage", False)

# Test and migrate gradually
feature_flags.set("use_new_storage", True)
feature_flags.set("use_new_features", True)

# Or enable all at once after testing
feature_flags.enable_all_new_features()
```

See the [Migration Guide](docs/Migration_Guide.md) for detailed instructions.

## 🏗️ Architecture

The refactored MDM uses a clean, modular architecture:

```
mdm/
├── interfaces/          # Abstract interfaces
├── core/               # New implementations
│   ├── storage/        # Storage backends
│   ├── features/       # Feature engineering
│   └── config/         # Configuration management
├── adapters/           # Legacy compatibility
├── performance/        # Optimization modules
└── cli/               # Command-line interface
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

# Run performance benchmarks
python -m mdm.testing.performance_benchmark
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/mdm/mdm-refactor-2025.git
cd mdm-refactor-2025

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
```

## 📊 Performance

The refactored version includes significant performance improvements:

- **Query Optimization**: Automatic query plan caching and optimization
- **Multi-Level Caching**: LRU cache for frequently accessed data
- **Batch Processing**: Efficient handling of large datasets
- **Connection Pooling**: Reuse database connections
- **Parallel Processing**: Multi-threaded feature generation

Benchmark results show 2-5x performance improvements for common operations.

## 🔒 Security

- SQL injection prevention through parameterized queries
- Path traversal protection for file operations
- Secure configuration with hidden sensitive values
- Audit logging for all operations

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original MDM contributors
- SQLAlchemy team for the excellent ORM
- Rich library for beautiful terminal output
- DuckDB team for the amazing analytics engine

## 📞 Support

- 📚 [Documentation](https://mdm.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/mdm/mdm-refactor/issues)
- 💬 [Discussions](https://github.com/mdm/mdm-refactor/discussions)
- 📧 [Email Support](mailto:support@mdm.io)

---

**Note**: This is the 2025 refactored version of MDM. For the legacy version, see the `legacy` branch.