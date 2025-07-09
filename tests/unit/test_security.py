"""Security tests for MDM - SQL injection, path traversal, etc."""

import pytest
from pathlib import Path
import tempfile

from mdm.api import MDMClient
from mdm.dataset.manager import DatasetManager
from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError


class TestSQLInjectionProtection:
    """Test SQL injection protection in dataset operations."""
    
    def test_sql_injection_in_dataset_name(self, test_config):
        """Test that SQL injection attempts in dataset names are blocked."""
        client = MDMClient(config=test_config)
        
        # Various SQL injection attempts
        malicious_names = [
            "'; DROP TABLE datasets; --",
            "dataset'; DELETE FROM sqlite_master; --",
            "test' OR '1'='1",
            "dataset'; UPDATE datasets SET name='hacked'; --",
            "test\\'; DROP TABLE datasets; --",
            "dataset`; DROP TABLE datasets; --",
            "test\"; DROP TABLE datasets; --",
            "1; SELECT * FROM datasets WHERE '1'='1",
            "dataset' UNION SELECT * FROM sqlite_master--",
        ]
        
        for malicious_name in malicious_names:
            # Should either sanitize or reject the name
            with pytest.raises((ValueError, DatasetError)):
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Create a simple CSV file
                    csv_path = Path(tmpdir) / "data.csv"
                    csv_path.write_text("id,value\n1,100\n2,200")
                    
                    client.register_dataset(
                        name=malicious_name,
                        dataset_path=str(tmpdir),
                    )
    
    def test_sql_injection_in_query_operations(self, test_config):
        """Test SQL injection in query operations."""
        client = MDMClient(config=test_config)
        
        # First register a normal dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            csv_path.write_text("id,value\n1,100\n2,200")
            
            client.register_dataset(
                name="test_dataset",
                dataset_path=str(tmpdir),
            )
            
            # Try SQL injection in query
            malicious_queries = [
                "SELECT * FROM train; DROP TABLE train; --",
                "SELECT * FROM train WHERE 1=1; DELETE FROM train; --",
                "SELECT * FROM train UNION SELECT * FROM sqlite_master--",
            ]
            
            for query in malicious_queries:
                # Query should be executed safely or rejected
                try:
                    result = client.query_dataset("test_dataset", query)
                    # If query succeeds, verify no damage was done
                    verify_result = client.query_dataset(
                        "test_dataset", 
                        "SELECT COUNT(*) as count FROM train"
                    )
                    assert verify_result['count'].iloc[0] == 2
                except Exception:
                    # Query was rejected - also good
                    pass
    
    def test_sql_injection_in_column_names(self, test_config):
        """Test SQL injection attempts in column names."""
        registrar = DatasetRegistrar()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with malicious column names
            csv_path = Path(tmpdir) / "train.csv"
            csv_path.write_text(
                "id,\"value'; DROP TABLE datasets; --\"\n1,100\n2,200"
            )
            
            # Should handle malicious column names safely
            dataset_info = registrar.register(
                name="test_dataset",
                path=Path(tmpdir),
                force=True
            )
            
            # Verify dataset was created safely
            assert dataset_info is not None
            assert dataset_info.name == "test_dataset"
    
    def test_path_traversal_protection(self, test_config):
        """Test protection against path traversal attacks."""
        client = MDMClient(config=test_config)
        
        # Path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "../../../../tmp/evil",
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, DatasetError, FileNotFoundError)):
                client.register_dataset(
                    name="test",
                    dataset_path=malicious_path,
                )


class TestParameterValidation:
    """Test parameter validation to prevent injection attacks."""
    
    def test_dataset_name_validation(self):
        """Test that dataset names are properly validated."""
        manager = DatasetManager()
        
        # Invalid names that should be rejected
        invalid_names = [
            "",  # Empty
            " ",  # Whitespace only
            "dataset name",  # Spaces
            "dataset/name",  # Slash
            "dataset\\name",  # Backslash
            "dataset;name",  # Semicolon
            "dataset'name",  # Quote
            "dataset\"name",  # Double quote
            "dataset`name",  # Backtick
            "dataset\nname",  # Newline
            "dataset\x00name",  # Null byte
            "." * 256,  # Too long
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises((ValueError, DatasetError)):
                manager.validate_dataset_name(invalid_name)
    
    def test_safe_dataset_names(self):
        """Test that safe dataset names are accepted."""
        manager = DatasetManager()
        
        # Valid names that should be accepted
        valid_names = [
            "dataset",
            "dataset_123",
            "dataset-name",
            "MyDataset",  # Will be normalized to lowercase
            "data_2024_01",
            "test-dataset-v2",
        ]
        
        for valid_name in valid_names:
            normalized = manager.validate_dataset_name(valid_name)
            assert normalized == valid_name.lower()
            assert all(c.isalnum() or c in "_-" for c in normalized)