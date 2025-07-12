# MDM - ML Data Manager Overview

## Introduction

MDM (ML Data Manager) is a powerful, enterprise-grade dataset management system designed specifically for machine learning workflows. Version 1.0.0 represents a mature, production-ready solution that simplifies dataset management, feature engineering, and ML pipeline integration.

## Core Concepts

### The Problem MDM Solves

Managing datasets for machine learning projects often involves:
- Scattered CSV/Parquet files across different directories
- Manual tracking of train/test splits
- Repetitive feature engineering code
- Inconsistent data preprocessing
- Difficulty in dataset versioning and discovery

MDM addresses these challenges by providing a centralized, database-backed system with automatic feature engineering and comprehensive management tools.

### Key Design Principles

1. **Simplicity First**: Easy to use CLI and Python API
2. **Local-First**: All data stored locally, no cloud dependencies
3. **Performance**: Optimized for datasets from MB to GB scale
4. **Extensibility**: Plugin system for custom features
5. **Database-Powered**: Leverages SQLAlchemy for robust data management

## Architecture Overview

### Two-Tier Database System

MDM uses a unique two-tier architecture:

```
┌─────────────────────────────────────────────────────┐
│                   MDM System                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Discovery Layer (YAML)          Data Layer (DB)    │
│  ┌─────────────────────┐      ┌──────────────────┐ │
│  │ ~/.mdm/config/      │      │ ~/.mdm/datasets/ │ │
│  │ └── datasets/       │      │ └── {name}/      │ │
│  │     ├── iris.yaml   │ ───► │     └── data.db  │ │
│  │     ├── titanic.yaml│      │                  │ │
│  │     └── sales.yaml  │      │                  │ │
│  └─────────────────────┘      └──────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Discovery Layer**: Lightweight YAML files containing dataset metadata and configuration
**Data Layer**: Full database (SQLite/DuckDB/PostgreSQL) containing actual data and features

### Component Architecture

```
MDM Application
├── CLI Interface (Typer + Rich)
│   ├── Dataset Commands
│   ├── Batch Operations
│   └── System Commands
│
├── Python API (MDMClient)
│   ├── Registration Client
│   ├── Query Client
│   ├── Export Client
│   └── ML Integration Client
│
├── Core Services
│   ├── Dataset Manager
│   ├── Feature Generator
│   ├── Storage Factory
│   └── Configuration Manager
│
├── Storage Backends
│   ├── SQLite (Default)
│   ├── DuckDB (Analytics)
│   └── PostgreSQL (Enterprise)
│
└── Feature Engineering
    ├── Generic Transformers
    └── Custom Transformers
```

### Data Flow

1. **Registration**: User provides path to raw data → MDM auto-detects structure → Creates database → Generates features
2. **Usage**: User queries via CLI/API → MDM loads from database → Returns processed data
3. **Export**: User requests export → MDM packages data → Outputs in requested format

## Key Features

### 1. Automatic Dataset Discovery

MDM intelligently detects dataset structures:
- **Kaggle Competitions**: Recognizes train.csv/test.csv patterns
- **Single Files**: Handles individual CSV/Parquet files
- **Multiple Files**: Supports complex directory structures
- **Compressed Data**: Automatically handles .gz files

### 2. Smart Feature Engineering

Two-tier feature generation system:
- **Generic Features**: Automatic generation based on column types
  - Statistical: Z-scores, outliers, log transforms
  - Temporal: Date components, cyclical features
  - Categorical: One-hot encoding, frequency encoding
  - Text: Length, word count, special characters
- **Custom Features**: Dataset-specific transformations via plugins

### 3. Backend Flexibility

Choose the right storage for your needs:
- **SQLite**: Zero configuration, perfect for small-medium datasets
- **DuckDB**: Columnar storage, optimized for analytics
- **PostgreSQL**: Multi-user access, enterprise features

### 4. Rich CLI Experience

Beautiful terminal interface with:
- Progress bars for long operations
- Formatted tables for data display
- Color-coded output for clarity
- Interactive confirmations for destructive operations

### 5. Comprehensive API

Full programmatic access for integration:
```python
from mdm import MDMClient

client = MDMClient()
df = client.load_dataset("titanic")
```

## Use Cases

### 1. Kaggle Competitions
```bash
# Download competition data
kaggle competitions download -c titanic

# Register with MDM
mdm dataset register titanic ./titanic --target Survived

# Load in notebook
df = client.load_dataset("titanic")
```

### 2. ML Pipeline Integration
```python
# Use MDM as data source for pipelines
class MLPipeline:
    def __init__(self, dataset_name):
        self.client = MDMClient()
        self.dataset = dataset_name
    
    def get_train_test(self):
        return self.client.ml.load_train_test_split(self.dataset)
```

### 3. Data Exploration
```bash
# Quick dataset overview
mdm dataset info sales_data

# View statistics
mdm dataset stats sales_data

# Export for analysis
mdm dataset export sales_data --format parquet
```

### 4. Feature Engineering Workflows
```python
# Create custom features
class CustomFeatures(BaseDomainFeatures):
    def get_domain_features(self, df):
        return {
            'revenue_per_user': df['revenue'] / df['users'],
            'is_weekend': df['date'].dt.dayofweek.isin([5, 6])
        }
```

## Performance Characteristics

MDM is optimized for:
- **Dataset Size**: MB to low GB range
- **Row Count**: Tested up to 10M rows
- **Column Count**: Handles 1000+ columns
- **Registration Speed**: ~1M rows/minute
- **Query Performance**: Sub-second for most operations

## System Requirements

- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2x dataset size for processing
- **OS**: Linux, macOS, Windows

## Next Steps

- [Installation Guide](02_Installation.md) - Get MDM up and running
- [Quick Start](03_Quick_Start.md) - Your first dataset in 5 minutes
- [CLI Reference](04_CLI_Reference.md) - Complete command documentation
- [Python API](05_Python_API.md) - Programmatic usage guide

## Design Philosophy

MDM follows these principles:

1. **Convention over Configuration**: Sensible defaults that just work
2. **Progressive Disclosure**: Simple tasks are simple, complex tasks are possible
3. **Fail-Safe**: Non-destructive by default, confirmations for dangerous operations
4. **Transparent**: Clear about what's happening under the hood
5. **Performant**: Optimized for common data science workflows

## Comparison with Alternatives

| Feature | MDM | Pandas | DVC | Feast |
|---------|-----|--------|-----|-------|
| Local-first | ✅ | ✅ | ✅ | ❌ |
| Auto Features | ✅ | ❌ | ❌ | ✅ |
| SQL Interface | ✅ | ❌ | ❌ | ✅ |
| Multiple Backends | ✅ | ❌ | ✅ | ❌ |
| Zero Config | ✅ | ✅ | ❌ | ❌ |
| Dataset Discovery | ✅ | ❌ | ✅ | ❌ |

MDM occupies a unique position: simpler than Feast, more structured than raw Pandas, and more feature-rich than DVC for ML-specific workflows.