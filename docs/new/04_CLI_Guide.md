# 4. CLI Guide

The `mdm` command-line interface provides a comprehensive set of tools for managing your datasets.

## Global Commands

### `mdm version`

Displays the installed version of MDM.

```bash
mdm version
```

### `mdm info`

Shows a summary of the system configuration, storage paths, database settings, and current status.

```bash
mdm info
```

## Dataset Management (`mdm dataset`)

The `dataset` subcommand is the primary entry point for all dataset-related operations.

### `mdm dataset register`

Registers a new dataset with MDM. This is the most critical command and has many options for customizing the registration process.

**Usage:**

```bash
mdm dataset register <name> <path> [OPTIONS]
```

**Arguments:**

*   `name`: The unique name for the dataset.
*   `path`: The path to the dataset directory or file.

**Common Options:**

*   `--target, -t`: The name of the target column for machine learning tasks.
*   `--problem-type`: The type of ML problem (e.g., `classification`, `regression`).
*   `--id-columns`: A comma-separated list of column names to be treated as identifiers.
*   `--description, -d`: A short description of the dataset.
*   `--tags`: A comma-separated list of tags for easy searching.
*   `--force, -f`: Forces re-registration if the dataset already exists.
*   `--no-features`: Skips the feature generation step.

### `mdm dataset list`

Lists all registered datasets.

**Usage:**

```bash
mdm dataset list [OPTIONS]
```

**Options:**

*   `--format, -f`: The output format (`rich`, `text`, `json`).
*   `--sort-by`: The field to sort by (`name`, `registration_date`).
*   `--limit`: The maximum number of datasets to display.

### `mdm dataset info`

Displays detailed information about a specific dataset.

**Usage:**

```bash
mdm dataset info <name>
```

### `mdm dataset stats`

Shows computed statistics for a dataset.

**Usage:**

```bash
mdm dataset stats <name> [OPTIONS]
```

**Options:**

*   `--full`: Shows all available statistics, including correlations.
*   `--export <path>`: Exports the statistics to a file.

### `mdm dataset update`

Updates the metadata of an existing dataset.

**Usage:**

```bash
mdm dataset update <name> [OPTIONS]
```

**Options:**

*   `--description`: A new description.
*   `--target`: A new target column.
*   `--problem-type`: A new problem type.
*   `--tags`: A new set of comma-separated tags.

### `mdm dataset export`

Exports a dataset's data to local files.

**Usage:**

```bash
mdm dataset export <name> [OPTIONS]
```

**Options:**

*   `--output-dir, -o`: The directory to save the exported files.
*   `--format, -f`: The output file format (`csv`, `parquet`, `json`).
*   `--table`: The specific table to export.

### `mdm dataset remove`

Removes a dataset from MDM, deleting its configuration and database.

**Usage:**

```bash
mdm dataset remove <name> [OPTIONS]
```

**Options:**

*   `--force, -f`: Skips the confirmation prompt.
*   `--dry-run`: Shows what would be deleted without actually deleting anything.

### `mdm dataset search`

Searches for datasets by name, description, or tags.

**Usage:**

```bash
mdm dataset search <query> [OPTIONS]
```

**Options:**

*   `--tag`: Searches for datasets with a specific tag.
*   `--deep`: Performs a slower, more thorough search in dataset metadata.
