# MDM Documentation

Welcome to the MDM (ML Data Manager) documentation. This guide covers everything you need to know about using MDM for managing your machine learning datasets.

## Documentation Structure

1. **[Introduction](00_Introduction.md)** - Overview and motivation
2. **[Features](01_Features.md)** - Comprehensive feature list
3. **[System Architecture](02_System_Architecture.md)** - Technical design and components
4. **[Configuration](03_Configuration.md)** - Setting up MDM
5. **[Dataset Registration](04_Dataset_Registration.md)** - How to register datasets
6. **[Dataset Management](05_Dataset_Management.md)** - Managing your datasets
7. **[Feature Engineering](06_Feature_Engineering.md)** - Automatic and custom features
8. **[CLI Usage](07_CLI_Usage.md)** - Command-line interface guide
9. **[Programmatic API](08_Programmatic_API.md)** - Python API reference
10. **[Advanced Features](09_Advanced_Features.md)** - Time series, performance, and more
11. **[Best Practices](10_Best_Practices.md)** - Recommendations and tips
12. **[Troubleshooting](11_Troubleshooting.md)** - Common issues and solutions
13. **[Summary](12_Summary.md)** - Key takeaways and quick reference
14. **[Testing and Validation](13_Testing_and_Validation.md)** - Testing guide

## Quick Start

```bash
# Install MDM
pip install mdm

# Register a dataset
mdm dataset register my_dataset /path/to/data

# Get dataset info
mdm dataset info my_dataset

# Use in Python
from mdm import load_dataset
train_df, test_df = load_dataset("my_dataset")
```

## Getting Help

- Check the [Troubleshooting](11_Troubleshooting.md) guide for common issues
- Review [Best Practices](10_Best_Practices.md) for optimal usage
- See the [Summary](12_Summary.md) for a quick overview

## Testing

After reading the documentation, you can verify MDM functionality using the test scripts:

```bash
# Quick test
./scripts/test_e2e_quick.sh test_dataset ./data/sample

# Full test
./scripts/test_e2e_nocolor.sh test_dataset ./data/sample

# Interactive demo
./scripts/test_e2e_demo.sh demo_dataset
```

## Contributing

MDM is designed to be extensible. You can:
- Add custom feature transformers
- Create new storage backends
- Extend the CLI with new commands
- Contribute to the documentation

## License

See LICENSE file in the repository root.