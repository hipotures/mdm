"""New PostgreSQL storage backend implementation.

This module provides the refactored PostgreSQL backend with improved design
and better separation of concerns.
"""
from typing import Any, Dict, Optional
from pathlib import Path
import logging
from urllib.parse import quote_plus

from sqlalchemy import Engine, text, create_engine
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from .base import NewStorageBackend
from mdm.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class NewPostgreSQLBackend(NewStorageBackend):
    """New PostgreSQL storage backend implementation.
    
    Key improvements over legacy:
    - Better connection pooling
    - Proper database/schema management
    - Improved error handling
    - Better security (SSL support)
    """
    
    @property
    def backend_type(self) -> str:
        """Get backend type identifier."""
        return "postgresql"
    
    def _create_connection_string(self, database_path: str) -> str:
        """Create PostgreSQL connection string.
        
        Args:
            database_path: Database name (for PostgreSQL)
            
        Returns:
            PostgreSQL connection string
        """
        # Build connection string from config
        user = self.config.get("user", "postgres")
        password = self.config.get("password", "")
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 5432)
        sslmode = self.config.get("sslmode", "prefer")
        
        # URL-encode password if present
        if password:
            password = quote_plus(password)
            auth = f"{user}:{password}"
        else:
            auth = user
        
        # Use database_path as database name
        database = database_path
        
        # Build connection string
        conn_string = f"postgresql://{auth}@{host}:{port}/{database}"
        
        # Add SSL mode
        if sslmode:
            conn_string += f"?sslmode={sslmode}"
        
        return conn_string
    
    def _get_engine_args(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific engine arguments."""
        return {
            "pool_size": self.config.get("pool_size", 10),
            "max_overflow": self.config.get("max_overflow", 20),
            "pool_timeout": self.config.get("pool_timeout", 30),
            "pool_recycle": self.config.get("pool_recycle", 3600),
            "pool_pre_ping": True,  # Verify connections before use
            "connect_args": {
                "connect_timeout": self.config.get("connect_timeout", 10),
                "application_name": "mdm_refactored",
            }
        }
    
    def _initialize_backend_specific(self, engine: Engine) -> None:
        """Perform PostgreSQL-specific initialization.
        
        Sets PostgreSQL-specific settings and creates extensions if needed.
        """
        with engine.connect() as conn:
            # Set session parameters
            session_params = {
                "statement_timeout": self.config.get("statement_timeout", 0),
                "lock_timeout": self.config.get("lock_timeout", 0),
                "idle_in_transaction_session_timeout": self.config.get(
                    "idle_in_transaction_session_timeout", 0
                ),
            }
            
            for param, value in session_params.items():
                if value > 0:
                    try:
                        conn.execute(text(f"SET {param} = {value}"))
                        logger.debug(f"Set PostgreSQL parameter {param} = {value}")
                    except Exception as e:
                        logger.warning(f"Failed to set parameter {param}: {e}")
            
            # Create commonly used extensions
            extensions = self.config.get("extensions", ["uuid-ossp", "pg_stat_statements"])
            for ext in extensions:
                try:
                    conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS \"{ext}\""))
                    logger.debug(f"Created PostgreSQL extension: {ext}")
                except Exception as e:
                    logger.warning(f"Failed to create extension {ext}: {e}")
    
    def database_exists(self, database_path: str) -> bool:
        """Check if database exists.
        
        Args:
            database_path: Database name
            
        Returns:
            True if database exists
        """
        # Connect to postgres database to check
        admin_conn_string = self._create_admin_connection_string()
        
        try:
            engine = create_engine(admin_conn_string, isolation_level="AUTOCOMMIT")
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                    {"dbname": database_path}
                )
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check if database exists: {e}")
            return False
        finally:
            engine.dispose()
    
    def create_database(self, database_path: str) -> None:
        """Create empty database.
        
        Args:
            database_path: Database name
        """
        if self.database_exists(database_path):
            logger.info(f"Database {database_path} already exists")
            return
        
        # Connect to postgres database to create new database
        admin_conn_string = self._create_admin_connection_string()
        
        try:
            # Use psycopg2 directly for database creation
            conn = psycopg2.connect(admin_conn_string)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                # Create database with UTF8 encoding
                cursor.execute(
                    f"CREATE DATABASE {database_path} "
                    f"ENCODING 'UTF8' "
                    f"LC_COLLATE 'en_US.UTF-8' "
                    f"LC_CTYPE 'en_US.UTF-8' "
                    f"TEMPLATE template0"
                )
            
            conn.close()
            logger.info(f"Created PostgreSQL database: {database_path}")
            
        except Exception as e:
            raise StorageError(f"Failed to create database: {e}")
    
    def drop_database(self, database_path: str) -> None:
        """Drop an existing database.
        
        Args:
            database_path: Database name
        """
        # Close any existing connections first
        matching_engines = [
            path for path in self._engines.keys() 
            if database_path in path
        ]
        for path in matching_engines:
            self._engines[path].dispose()
            del self._engines[path]
        
        if not self.database_exists(database_path):
            logger.info(f"Database {database_path} does not exist")
            return
        
        # Connect to postgres database to drop the target database
        admin_conn_string = self._create_admin_connection_string()
        
        try:
            # Use psycopg2 directly for database dropping
            conn = psycopg2.connect(admin_conn_string)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                # Terminate existing connections
                cursor.execute(
                    f"SELECT pg_terminate_backend(pid) "
                    f"FROM pg_stat_activity "
                    f"WHERE datname = '{database_path}' AND pid <> pg_backend_pid()"
                )
                
                # Drop database
                cursor.execute(f"DROP DATABASE IF EXISTS {database_path}")
            
            conn.close()
            logger.info(f"Dropped PostgreSQL database: {database_path}")
            
        except Exception as e:
            raise StorageError(f"Failed to drop database: {e}")
    
    def _get_file_extension(self) -> str:
        """Get file extension for database files.
        
        PostgreSQL doesn't use file extensions, return empty string.
        """
        return ""
    
    def _create_admin_connection_string(self) -> str:
        """Create connection string for administrative tasks.
        
        Returns:
            Connection string to postgres database
        """
        # Build connection string to postgres database
        user = self.config.get("user", "postgres")
        password = self.config.get("password", "")
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 5432)
        
        if password:
            password = quote_plus(password)
            auth = f"{user}:{password}"
        else:
            auth = user
        
        return f"postgresql://{auth}@{host}:{port}/postgres"
    
    def _get_dataset_path(self, dataset_name: str, base_path: Path) -> str:
        """Get path for dataset database.
        
        For PostgreSQL, this returns the database name with prefix.
        """
        prefix = self.config.get("database_prefix", "mdm_")
        return f"{prefix}{dataset_name}"
    
    def optimize_database(self, engine: Engine) -> None:
        """Run PostgreSQL-specific optimizations.
        
        Args:
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            # Get all tables
            tables = self.get_table_names(engine)
            
            for table in tables:
                try:
                    # Run VACUUM ANALYZE on each table
                    conn.execute(text(f"VACUUM ANALYZE {table}"))
                    logger.debug(f"Vacuumed and analyzed table {table}")
                    
                    # Reindex if needed
                    if self.config.get("reindex_on_optimize", False):
                        conn.execute(text(f"REINDEX TABLE {table}"))
                        logger.debug(f"Reindexed table {table}")
                        
                except Exception as e:
                    logger.warning(f"Failed to optimize table {table}: {e}")
            
            logger.info("Optimized PostgreSQL database")
    
    def get_database_size(self, database_path: str) -> int:
        """Get database size in bytes.
        
        Args:
            database_path: Database name
            
        Returns:
            Size in bytes
        """
        try:
            engine = self.get_engine(database_path)
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT pg_database_size(current_database())")
                )
                return result.scalar()
        except Exception as e:
            logger.error(f"Failed to get database size: {e}")
            return 0
    
    def create_schema(self, schema_name: str, engine: Engine) -> None:
        """Create a schema in the database.
        
        Args:
            schema_name: Name of schema to create
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            logger.info(f"Created schema: {schema_name}")
    
    def grant_permissions(
        self, 
        user: str, 
        permissions: str, 
        database: str,
        engine: Engine
    ) -> None:
        """Grant permissions to a user.
        
        Args:
            user: Username
            permissions: Permissions to grant (e.g., 'ALL', 'SELECT')
            database: Database name
            engine: SQLAlchemy engine
        """
        with engine.connect() as conn:
            conn.execute(
                text(f"GRANT {permissions} ON DATABASE {database} TO {user}")
            )
            logger.info(f"Granted {permissions} on {database} to {user}")