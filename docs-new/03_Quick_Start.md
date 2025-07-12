# Quick Start Guide

Get up and running with MDM in 5 minutes! This guide will walk you through registering your first dataset and performing common operations.

## Prerequisites

- MDM installed (see [Installation Guide](02_Installation.md))
- A CSV file or dataset directory ready to import

## Your First Dataset

### Step 1: Prepare Your Data

MDM works with various data formats. For this example, let's create a simple CSV file:

```bash
# Create a sample dataset
cat > sales_data.csv << 'EOF'
order_id,date,product,category,price,quantity,customer_id
1,2024-01-15,Laptop,Electronics,999.99,1,101
2,2024-01-15,Mouse,Electronics,29.99,2,102
3,2024-01-16,Notebook,Office,4.99,10,101
4,2024-01-16,Desk Chair,Furniture,199.99,1,103
5,2024-01-17,Monitor,Electronics,299.99,1,102
6,2024-01-17,Pen Set,Office,12.99,3,104
7,2024-01-18,Laptop,Electronics,999.99,1,105
8,2024-01-18,Keyboard,Electronics,79.99,1,105
EOF
```

### Step 2: Register the Dataset

```bash
# Basic registration
mdm dataset register sales sales_data.csv

# With target column for ML
mdm dataset register sales sales_data.csv --target price

# With additional metadata
mdm dataset register sales sales_data.csv \
    --target price \
    --id-columns order_id,customer_id \
    --description "Q1 2024 sales data" \
    --tags "sales,2024,quarterly"
```

You'll see output like:
```
✓ Validating dataset name...
✓ Checking path...
✓ Detecting dataset structure...
✓ Creating storage backend...
✓ Loading data...
  Loading sales_data.csv: 100% |████████████| 8 rows [00:00.1s]
✓ Detecting column types...
✓ Generating features...
  Statistical features: 12 generated
  Temporal features: 6 generated
  Categorical features: 8 generated
✓ Computing statistics...
✓ Saving configuration...

Dataset 'sales' registered successfully!
  • Location: ~/.mdm/datasets/sales/
  • Rows: 8
  • Columns: 7 (+ 26 features)
  • Size: 45.2 KB
```

### Step 3: Explore Your Dataset

#### View Dataset Information
```bash
mdm dataset info sales
```

Output:
```
Dataset: sales
═══════════════════════════════════════════════════════════
  Path: ~/.mdm/datasets/sales/data.db
  Backend: sqlite
  Created: 2024-01-20 10:30:15
  
Data Structure:
  • Tables: train (8 rows)
  • Original columns: 7
  • Feature columns: 26
  • Target: price
  • ID columns: order_id, customer_id
  
Storage:
  • Database size: 45.2 KB
  • Compression: none
```

#### View Statistics
```bash
mdm dataset stats sales
```

Output:
```
Statistics for 'sales'
═══════════════════════════════════════════════════════════

Numeric Columns:
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Column     ┃ Mean   ┃ Std    ┃ Min    ┃ Max     ┃ Nulls ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ price      │ 327.48 │ 420.15 │ 4.99   │ 999.99  │ 0     │
│ quantity   │ 2.5    │ 3.16   │ 1      │ 10      │ 0     │
│ order_id   │ 4.5    │ 2.45   │ 1      │ 8       │ 0     │
└────────────┴────────┴────────┴────────┴─────────┴───────┘

Categorical Columns:
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Column     ┃ Unique       ┃ Most Common            ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ product    │ 6            │ Laptop (2)             │
│ category   │ 3            │ Electronics (5)        │
└────────────┴──────────────┴────────────────────────┘
```

### Step 4: Use Your Dataset

#### CLI Export
```bash
# Export as Parquet
mdm dataset export sales --format parquet

# Export with compression
mdm dataset export sales --format csv --compression gzip

# Export to specific location
mdm dataset export sales --output ~/exports/sales_processed.csv
```

#### Python API Usage
```python
from mdm import MDMClient

# Initialize client
client = MDMClient()

# Load dataset
df = client.load_dataset("sales")
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Access original and feature columns
print("\nOriginal columns:", df.columns[:7].tolist())
print("Feature columns:", df.columns[7:].tolist()[:5], "...")

# Get dataset metadata
info = client.get_dataset_info("sales")
print(f"\nTarget column: {info.target_column}")
print(f"Problem type: {info.problem_type}")
```

## Working with Kaggle Datasets

MDM has special support for Kaggle competition formats:

```bash
# Download Kaggle competition
kaggle competitions download -c titanic
unzip titanic.zip -d titanic/

# Register with auto-detection
mdm dataset register titanic ./titanic --target Survived

# MDM automatically detects:
# - train.csv as training data
# - test.csv as test data
# - sample_submission.csv format
```

## Advanced Registration Options

### Multiple File Datasets
```bash
# Directory with multiple CSVs
mdm dataset register multi_year ./data/yearly/
# MDM will find: 2022.csv, 2023.csv, 2024.csv
```

### Compressed Files
```bash
# Gzipped CSV
mdm dataset register compressed data.csv.gz
```

### Force Column Types
```bash
mdm dataset register typed_data data.csv \
    --categorical-columns user_type,region \
    --datetime-columns created_at,updated_at \
    --text-columns description,notes
```

### Skip Feature Generation
```bash
# Faster registration without features
mdm dataset register quick_data data.csv --no-features
```

## Dataset Management

### List All Datasets
```bash
mdm dataset list
```

Output:
```
Datasets (Backend: sqlite)
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name     ┃ Created            ┃ Rows   ┃ Size      ┃ Backend  ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ sales    │ 2024-01-20 10:30   │ 8      │ 45.2 KB   │ ✓        │
│ titanic  │ 2024-01-19 15:45   │ 891    │ 1.2 MB    │ ✓        │
└──────────┴────────────────────┴────────┴───────────┴──────────┘
```

### Search Datasets
```bash
# By name pattern
mdm dataset search sales

# By tag
mdm dataset search --tag classification

# Combined
mdm dataset search ti --tag kaggle
```

### Update Dataset Metadata
```bash
# Add description
mdm dataset update sales --description "Updated Q1 sales including returns"

# Add tags
mdm dataset update sales --tags "sales,q1,2024,validated"

# Change target
mdm dataset update sales --target quantity
```

### Remove Datasets
```bash
# With confirmation
mdm dataset remove sales

# Force removal
mdm dataset remove sales --force

# Dry run to see what would be removed
mdm dataset remove sales --dry-run
```

## Batch Operations

Work with multiple datasets efficiently:

```bash
# Export multiple datasets
mdm batch export sales,titanic --format parquet

# Get stats for all datasets
mdm batch stats "*"

# Remove datasets matching pattern
mdm batch remove "test_*" --dry-run
```

## Best Practices

### 1. Naming Conventions
```bash
# Good names
mdm dataset register sales_2024_q1 ./data/sales_q1.csv
mdm dataset register customer_churn ./churn_data.csv
mdm dataset register kaggle_titanic ./titanic/

# Avoid
mdm dataset register data ./data.csv  # Too generic
mdm dataset register 2024 ./2024.csv  # Starts with number
```

### 2. Use Tags Effectively
```bash
# Tag by purpose
--tags "training,production"

# Tag by date
--tags "2024-q1,january"

# Tag by status
--tags "validated,clean"

# Tag by source
--tags "kaggle,competition"
```

### 3. ID Columns Matter
```bash
# Always specify ID columns for:
# - Joining datasets
# - Creating submissions
# - Tracking entities

mdm dataset register customers data.csv \
    --id-columns customer_id \
    --target lifetime_value
```

### 4. Problem Type Helps Features
```bash
# Specify problem type for better features
--problem-type classification  # Binary/multiclass
--problem-type regression      # Continuous target
--problem-type clustering      # No target
--problem-type time-series     # Temporal prediction
```

## Common Workflows

### ML Competition Workflow
```bash
# 1. Register competition data
mdm dataset register competition ./kaggle_data --target target

# 2. Explore in notebook
from mdm import MDMClient
client = MDMClient()
train = client.load_dataset("competition", table="train")
test = client.load_dataset("competition", table="test")

# 3. Create submission
predictions = model.predict(test)
client.ml.create_submission(
    "competition", 
    predictions,
    "submission.csv"
)
```

### Feature Engineering Workflow
```bash
# 1. Register without features
mdm dataset register raw_data data.csv --no-features

# 2. Create custom features
cat > ~/.mdm/config/custom_features/raw_data.py << 'EOF'
from mdm.features.custom.base import BaseDomainFeatures

class CustomFeatureOperations(BaseDomainFeatures):
    def get_domain_features(self, df):
        return {
            'price_per_unit': df['total_price'] / df['quantity'],
            'is_bulk_order': (df['quantity'] > 10).astype(int)
        }
EOF

# 3. Re-register with features
mdm dataset register raw_data data.csv --force
```

### Data Versioning Workflow
```bash
# Simple versioning through naming
mdm dataset register sales_v1 ./sales_jan.csv
mdm dataset register sales_v2 ./sales_jan_cleaned.csv
mdm dataset register sales_v3 ./sales_jan_final.csv

# List versions
mdm dataset list | grep sales_v
```

## Next Steps

Now that you've mastered the basics:

- [CLI Reference](04_CLI_Reference.md) - Full command documentation
- [Python API Guide](05_Python_API.md) - Advanced programmatic usage
- [Feature Engineering](07_Feature_Engineering.md) - Custom transformations
- [ML Integration](10_ML_Integration.md) - Framework integration

## Quick Reference Card

```bash
# Essential Commands
mdm dataset register <name> <path>    # Register dataset
mdm dataset list                      # List all datasets
mdm dataset info <name>               # Show dataset details
mdm dataset stats <name>              # Show statistics
mdm dataset export <name>             # Export dataset
mdm dataset remove <name>             # Remove dataset

# Key Options
--target <column>                     # Specify target column
--id-columns <col1,col2>             # Specify ID columns
--tags <tag1,tag2>                   # Add tags
--force                              # Overwrite existing
--no-features                        # Skip feature generation

# Python Quickstart
from mdm import MDMClient
client = MDMClient()
df = client.load_dataset("name")
```