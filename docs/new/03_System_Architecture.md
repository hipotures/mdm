# 3. System Architecture

## Core Architecture

MDM employs a two-tier database system:

1.  **Dataset-Specific Databases**: Each registered dataset is stored in its own dedicated database located in `~/.mdm/datasets/{dataset_name}/`. This decentralized approach ensures data isolation and makes datasets portable. MDM supports SQLite, DuckDB, and PostgreSQL as database backends.
2.  **Discovery Mechanism**: A set of YAML configuration files in `~/.mdm/config/datasets/` act as lightweight pointers to the datasets. These files contain the metadata for each dataset, allowing MDM to discover and manage them.

### Single Backend Principle

A critical design principle in MDM is that it uses **only one type of database backend for all datasets at any given time**.

*   The active backend is determined by the `database.default_backend` setting in your `~/.mdm/mdm.yaml` configuration file.
*   When you switch the backend (e.g., from `sqlite` to `duckdb`), only the datasets created with the currently active backend will be visible and accessible.
*   There is no CLI flag to override the backend for a specific command. To use a different backend, you must modify the configuration file *before* registering any datasets with that backend.

## Key Modules

The `src/mdm` directory contains the core logic of the application, organized into the following modules:

*   `api/`: Provides the high-level programmatic `MDMClient` for interacting with MDM from Python code.
*   `cli/`: Implements the command-line interface using Typer and Rich for formatted output.
*   `config/`: Manages application settings using Pydantic, handling configuration from defaults, YAML files, and environment variables.
*   `core/`: Contains fundamental components like custom exceptions and logging setup.
*   `dataset/`: Holds the primary business logic for dataset management, including registration, modification, and removal.
*   `features/`: Implements the feature engineering system, including generic and custom feature transformers.
*   `interfaces/`: Defines abstract base classes and interfaces that ensure components adhere to specific contracts.
*   `models/`: Contains Pydantic models for data structures used throughout the application, such as dataset metadata.
*   `monitoring/`: Provides simple capabilities for monitoring and collecting metrics.
*   `performance/`: Includes tools and settings related to performance optimization.
*   `services/`: Contains service classes that orchestrate complex operations.
*   `storage/`: Implements the storage backend logic for different databases (SQLite, DuckDB, PostgreSQL).
*   `utils/`: A collection of utility functions used across the application.

## Dataset Registration Process

The dataset registration is a comprehensive 12-step process:

1.  Validate the provided dataset name.
2.  Check if a dataset with the same name already exists.
3.  Validate the file path to the dataset.
4.  Auto-detect the dataset's structure (e.g., Kaggle format, directory of CSVs).
5.  Discover all relevant data files.
6.  Attempt to detect ID columns and a potential target column.
7.  Create the physical storage backend (database).
8.  Load the data in batches (defaulting to 10,000 rows per chunk) for memory efficiency.
9.  Detect column types and schemas.
10. Generate a set of default features.
11. Compute summary statistics for the dataset.
12. Save the final dataset configuration to a YAML file.
