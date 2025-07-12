# 6. Dataset Management

This section provides a deeper dive into how MDM handles datasets, from registration to removal.

## The Dataset Lifecycle

A dataset in MDM goes through several stages:

1.  **Registration**: A new dataset is created using `mdm dataset register` or `client.register_dataset()`. During this stage, MDM analyzes the data, creates a dedicated database, computes metadata and statistics, and generates features.
2.  **Usage**: Once registered, the dataset can be loaded into memory for analysis and model training, queried using SQL, and exported to various formats.
3.  **Update**: The dataset's metadata (like its description, tags, or target column) can be updated as needed.
4.  **Removal**: When a dataset is no longer needed, it can be removed from MDM, which deletes its database and configuration.

## Dataset Structure

Each registered dataset consists of two main components:

*   **Database**: A dedicated database (SQLite, DuckDB, or PostgreSQL) that stores the actual data. This database is located in `~/.mdm/datasets/{dataset_name}/`.
*   **Configuration File**: A YAML file located in `~/.mdm/config/datasets/{dataset_name}.yml` that contains all the metadata about the dataset, such as its schema, statistics, and user-defined properties.

This separation of data and metadata allows for a flexible and robust system.

## Data Loading and Caching

MDM is designed to be memory-efficient. When you register a dataset, the data is loaded in chunks (batches) to avoid overwhelming your system's memory. The default batch size is 10,000 rows, but this can be configured.

When you load a dataset for use in your code, MDM can cache the data to speed up subsequent loads.
