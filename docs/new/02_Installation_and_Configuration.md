# 2. Installation and Configuration

## Installation

It is recommended to use `uv` for dependency management.

```bash
# Install uv, create a virtual environment, and activate it
pip install uv
uv venv
source .venv/bin/activate

# Install the project in editable mode
uv pip install -e .
```

## Configuration

MDM uses a hierarchical configuration system. The settings are loaded in the following order of precedence:

1.  **Defaults**: Hardcoded default values.
2.  **YAML File**: A YAML file located at `~/.mdm/mdm.yaml`.
3.  **Environment Variables**: The highest priority, overriding all other settings.

### YAML Configuration

Here is an example of the `mdm.yaml` file:

```yaml
database:
  default_backend: "sqlite"
  backends:
    sqlite:
      synchronous: "NORMAL"
    duckdb:
      memory_limit: "1GB"
performance:
  batch_size: 10000
logging:
  level: "INFO"
  file: null
```

### Environment Variables

Environment variables must be prefixed with `MDM_`. Nested keys are separated by double underscores.

*   `MDM_DATABASE_DEFAULT_BACKEND=duckdb`
*   `MDM_PERFORMANCE_BATCH_SIZE=5000`
*   `MDM_LOGGING_LEVEL=DEBUG`
*   `MDM_LOGGING_FILE=/var/log/mdm.log`
