# 1. Project Overview

MDM (ML Data Manager) is a standalone, enterprise-grade dataset management system for machine learning. It uses a decentralized architecture where each dataset is self-contained with its own database, supporting SQLite, DuckDB, and PostgreSQL backends via SQLAlchemy.

## Key Features

*   **Decentralized Architecture**: Each dataset has its own dedicated database, ensuring isolation and portability.
*   **Multiple Backends**: Supports SQLite, DuckDB, and PostgreSQL.
*   **CLI and Programmatic API**: Manage datasets through a powerful command-line interface or integrate MDM into your Python applications.
*   **Automated Metadata**: Automatically detects schemas, computes statistics, and versions datasets.
*   **Feature Engineering**: A flexible system for creating and managing features.
