# Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM
- **Storage**: 500MB for MDM + 2x your dataset size
- **OS**: Linux, macOS, or Windows 10+

### Recommended Setup
- **Python**: 3.11 or 3.12 (best performance)
- **Memory**: 8GB+ RAM
- **Storage**: SSD with 10GB+ free space
- **OS**: Linux (Ubuntu 20.04+) or macOS

## Installation Methods

### 1. Using pip (Stable Release)

```bash
# Install from PyPI
pip install mdm-ml

# Verify installation
mdm version
```

### 2. Using uv (Recommended - Faster)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that MDM uses internally:

```bash
# Install uv first
pip install uv

# Install MDM using uv
uv pip install mdm-ml

# Verify installation
mdm version
```

### 3. Development Installation (Latest Features)

```bash
# Clone the repository
git clone https://github.com/hipotures/mdm.git
cd mdm

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Verify installation
mdm version
```

### 4. System Package Managers

#### macOS (Homebrew)
```bash
# Coming soon
brew install mdm
```

#### Linux (snap)
```bash
# Coming soon
snap install mdm
```

## Post-Installation Setup

### 1. Verify Installation

```bash
# Check version
mdm version
# Expected output: MDM version 1.0.0

# Check system info
mdm info
```

### 2. Initial Configuration

MDM works out-of-the-box with sensible defaults. The configuration file is created automatically at `~/.mdm/mdm.yaml` on first use.

To customize settings before first use:

```bash
# Create config directory
mkdir -p ~/.mdm

# Create config file with custom settings
cat > ~/.mdm/mdm.yaml << 'EOF'
database:
  default_backend: sqlite  # or duckdb, postgresql
  
performance:
  batch_size: 10000
  max_workers: 4
  
logging:
  level: INFO
  file: ~/.mdm/logs/mdm.log
EOF
```

### 3. Backend-Specific Setup

#### SQLite (Default)
No additional setup required! SQLite is embedded and ready to use.

#### DuckDB
```bash
# DuckDB is automatically installed with MDM
# Just change the backend in config:
echo "database:
  default_backend: duckdb" > ~/.mdm/mdm.yaml
```

#### PostgreSQL
```bash
# Install PostgreSQL client libraries
# Ubuntu/Debian:
sudo apt-get install libpq-dev

# macOS:
brew install postgresql

# Install Python PostgreSQL adapter
pip install psycopg2-binary

# Configure connection in ~/.mdm/mdm.yaml
cat >> ~/.mdm/mdm.yaml << 'EOF'
database:
  default_backend: postgresql
  postgresql:
    host: localhost
    port: 5432
    user: mdm_user
    password: your_password
    database: mdm_db
EOF
```

## Environment Variables

MDM can be configured using environment variables, which override config file settings:

```bash
# Set default backend
export MDM_DATABASE_DEFAULT_BACKEND=duckdb

# Set batch size for large datasets
export MDM_PERFORMANCE_BATCH_SIZE=50000

# Enable debug logging
export MDM_LOGGING_LEVEL=DEBUG

# Set custom data directory
export MDM_PATHS_BASE_PATH=/data/mdm
```

## Docker Installation (Optional)

For isolated environments:

```bash
# Pull the official image
docker pull hipotures/mdm:latest

# Run with volume mount
docker run -it -v ~/.mdm:/root/.mdm hipotures/mdm mdm version

# Create alias for convenience
alias mdm='docker run -it -v ~/.mdm:/root/.mdm -v $(pwd):/data hipotures/mdm mdm'
```

## Troubleshooting Installation

### Common Issues

#### 1. Python Version Error
```
ERROR: Python 3.9+ required
```
**Solution**: Update Python or use pyenv:
```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11.8
pyenv global 3.11.8
```

#### 2. Permission Denied
```
ERROR: Permission denied: '/usr/local/lib/python3.x/...'
```
**Solution**: Use `--user` flag or virtual environment:
```bash
# User installation
pip install --user mdm-ml

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install mdm-ml
```

#### 3. Missing Dependencies
```
ERROR: Microsoft Visual C++ 14.0 is required (Windows)
```
**Solution**: Install Visual Studio Build Tools:
- Download from [Microsoft](https://visualstudio.microsoft.com/downloads/)
- Select "C++ build tools" workload

#### 4. DuckDB Import Error
```
ImportError: cannot import name 'duckdb'
```
**Solution**: Reinstall with proper dependencies:
```bash
pip uninstall mdm-ml duckdb
pip install mdm-ml[duckdb]
```

### Platform-Specific Notes

#### macOS Apple Silicon (M1/M2)
```bash
# Some dependencies may need Rosetta
softwareupdate --install-rosetta

# Or use arch-specific pip
arch -arm64 pip install mdm-ml
```

#### Windows WSL2
Recommended to use WSL2 for best compatibility:
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install python3-pip python3-venv
pip install mdm-ml
```

#### Linux Distributions

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3-pip python3-venv python3-dev
```

**Fedora/RHEL**:
```bash
sudo dnf install python3-pip python3-devel
```

**Arch Linux**:
```bash
sudo pacman -S python-pip
```

## Verification Tests

After installation, run these tests to ensure everything works:

```bash
# 1. CLI responds
mdm version

# 2. Can access help
mdm --help

# 3. Configuration is readable
mdm info

# 4. Can create a test dataset
echo "id,value
1,100
2,200" > test.csv

mdm dataset register test_dataset test.csv
mdm dataset info test_dataset
mdm dataset remove test_dataset --force

# 5. Python API works
python -c "from mdm import MDMClient; print('MDM Python API OK')"
```

## Upgrading MDM

### From pip
```bash
pip install --upgrade mdm-ml
```

### From Development
```bash
cd /path/to/mdm
git pull origin main
pip install -e ".[dev]"
```

### Breaking Changes
Check [CHANGELOG.md](https://github.com/hipotures/mdm/blob/main/CHANGELOG.md) before upgrading between major versions.

## Uninstallation

To completely remove MDM:

```bash
# Uninstall package
pip uninstall mdm-ml

# Remove data and config (optional)
rm -rf ~/.mdm

# Remove any aliases
unalias mdm  # If you created one
```

## Next Steps

- [Quick Start Guide](03_Quick_Start.md) - Register your first dataset
- [CLI Reference](04_CLI_Reference.md) - Explore available commands
- [Configuration](08_Configuration.md) - Customize MDM settings