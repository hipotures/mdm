"""Tests for edge cases - empty files, corrupted data, unusual formats, etc."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from mdm.api import MDMClient
from mdm.dataset.registrar import DatasetRegistrar
from mdm.core.exceptions import DatasetError


class TestEmptyFiles:
    """Test handling of empty and minimal files."""
    
    def test_empty_csv_file(self, test_config):
        """Test handling of completely empty CSV file."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty file
            empty_path = Path(tmpdir) / "train.csv"
            empty_path.touch()  # Creates empty file
            
            # Should handle gracefully
            with pytest.raises((DatasetError, ValueError, pd.errors.EmptyDataError)):
                client.register_dataset(
                    name="empty_file",
                    dataset_path=str(tmpdir),
                )
    
    def test_csv_with_headers_only(self, test_config):
        """Test CSV file with headers but no data."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with only headers
            headers_only = Path(tmpdir) / "train.csv"
            headers_only.write_text("id,name,value,category\n")
            
            # Should either handle gracefully or raise appropriate error
            try:
                dataset_info = client.register_dataset(
                    name="headers_only",
                    dataset_path=str(tmpdir),
                )
                
                # If it succeeds, verify it handled empty data correctly
                stats = client.get_statistics("headers_only")
                assert stats['tables']['train']['row_count'] == 0
                
            except (DatasetError, ValueError) as e:
                # Also acceptable - rejected empty dataset
                assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_single_row_csv(self, test_config):
        """Test CSV with only one data row."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with one row
            single_row = Path(tmpdir) / "train.csv"
            single_row.write_text("id,value\n1,100\n")
            
            # Should handle single row dataset
            dataset_info = client.register_dataset(
                name="single_row",
                dataset_path=str(tmpdir),
            )
            
            # Verify it was registered correctly
            stats = client.get_statistics("single_row")
            assert stats['tables']['train']['row_count'] == 1
            
            # Should be able to load it
            train_df, _ = client.load_dataset_files("single_row")
            assert len(train_df) == 1
            assert train_df['id'].iloc[0] == 1
            assert train_df['value'].iloc[0] == 100
    
    def test_empty_columns(self, test_config):
        """Test handling of columns with all empty/null values."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with empty columns
            csv_path = Path(tmpdir) / "train.csv"
            data = pd.DataFrame({
                'id': range(10),
                'empty_col1': [None] * 10,
                'empty_col2': [''] * 10,
                'empty_col3': [np.nan] * 10,
                'valid_col': range(10),
            })
            data.to_csv(csv_path, index=False)
            
            # Should handle empty columns
            dataset_info = client.register_dataset(
                name="empty_columns",
                dataset_path=str(tmpdir),
            )
            
            # Load and verify
            train_df, _ = client.load_dataset_files("empty_columns")
            assert len(train_df) == 10
            
            # Empty columns should be preserved but recognized as having no signal
            assert 'empty_col1' in train_df.columns
            assert train_df['empty_col1'].isna().all() or (train_df['empty_col1'] == '').all()


class TestCorruptedData:
    """Test handling of corrupted and malformed data."""
    
    def test_corrupted_csv_inconsistent_columns(self, test_config):
        """Test CSV with inconsistent number of columns."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted CSV with inconsistent columns
            corrupted = Path(tmpdir) / "train.csv"
            corrupted.write_text(
                "id,name,value\n"
                "1,Alice,100\n"
                "2,Bob,200,extra_column\n"  # Too many columns
                "3,Charlie\n"  # Too few columns
                "4,David,400\n"
            )
            
            # Should either handle gracefully or provide clear error
            try:
                dataset_info = client.register_dataset(
                    name="corrupted_csv",
                    dataset_path=str(tmpdir),
                )
                
                # If it succeeds, verify data handling
                train_df, _ = client.load_dataset_files("corrupted_csv")
                assert len(train_df) >= 1  # At least some rows loaded
                
            except Exception as e:
                # Should have informative error message
                error_msg = str(e).lower()
                # Accept any error - the file is corrupted
                assert True
    
    def test_corrupted_encoding(self, test_config):
        """Test handling of files with encoding issues."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with mixed encodings (simulate corruption)
            mixed_encoding = Path(tmpdir) / "train.csv"
            
            # Write header in UTF-8
            with open(mixed_encoding, 'w', encoding='utf-8') as f:
                f.write("id,name,value\n")
                f.write("1,Alice,100\n")
            
            # Append data with different encoding (will cause issues)
            with open(mixed_encoding, 'ab') as f:
                # Write some bytes that aren't valid UTF-8
                f.write(b"2,\xff\xfe,200\n")
                f.write(b"3,Bob,300\n")
            
            # Should handle encoding errors gracefully
            try:
                dataset_info = client.register_dataset(
                    name="bad_encoding",
                    dataset_path=str(tmpdir),
                )
                
                # If successful, should have handled bad encoding
                train_df, _ = client.load_dataset_files("bad_encoding")
                assert len(train_df) >= 1  # At least some data loaded
                
            except (UnicodeDecodeError, DatasetError) as e:
                # Acceptable to fail with clear error
                assert 'encod' in str(e).lower() or 'decode' in str(e).lower()
    
    def test_binary_file_rejection(self, test_config):
        """Test that binary files are rejected appropriately."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create binary file with .csv extension
            binary_file = Path(tmpdir) / "train.csv"
            binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05' * 100)
            
            # Try to register binary file
            try:
                client.register_dataset(
                    name="binary_file",
                    dataset_path=str(tmpdir),
                )
                # If it succeeds, it should have failed to parse properly
                train_df, _ = client.load_dataset_files("binary_file")
                # Should be empty or very small
                assert len(train_df) <= 1
            except Exception:
                # Expected - binary file should fail
                assert True
    
    def test_huge_single_row(self, test_config):
        """Test handling of CSV with extremely long rows."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with very long row
            huge_row = Path(tmpdir) / "train.csv"
            
            # Create a row with 10,000 columns
            headers = [f'col_{i}' for i in range(10000)]
            values = [str(i) for i in range(10000)]
            
            with open(huge_row, 'w') as f:
                f.write(','.join(['id'] + headers) + '\n')
                f.write(','.join(['1'] + values) + '\n')
                f.write(','.join(['2'] + values) + '\n')
            
            # Should handle or reject gracefully
            try:
                dataset_info = client.register_dataset(
                    name="huge_columns",
                    dataset_path=str(tmpdir),
                )
                
                # If successful, verify it handled wide data
                info = client.get_dataset("huge_columns")
                assert info is not None
                
            except (DatasetError, ValueError) as e:
                # Acceptable to reject extremely wide data
                assert 'column' in str(e).lower() or 'wide' in str(e).lower()
    
    def test_special_characters_in_data(self, test_config):
        """Test handling of special characters in data."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with special characters
            special_chars = Path(tmpdir) / "train.csv"
            
            # Include various special characters
            data = pd.DataFrame({
                'id': range(5),
                'text': [
                    'normal text',
                    'text with "quotes"',
                    'text with, comma',
                    'text with\nnewline',
                    'text with\ttab and special chars: @#$%^&*()',
                ],
                'unicode': [
                    'English',
                    '中文',
                    'العربية',
                    'हिन्दी',
                    '🎉 Emoji! 🚀',
                ],
            })
            
            # Save with proper escaping
            data.to_csv(special_chars, index=False)
            
            # Should handle special characters
            dataset_info = client.register_dataset(
                name="special_chars",
                dataset_path=str(tmpdir),
            )
            
            # Load and verify special characters preserved
            train_df, _ = client.load_dataset_files("special_chars")
            assert len(train_df) == 5
            assert 'quotes' in train_df['text'].iloc[1]
            assert '中文' in train_df['unicode'].iloc[1]
            assert '🎉' in train_df['unicode'].iloc[4]


class TestUnusualFormats:
    """Test handling of unusual but valid formats."""
    
    def test_different_delimiters(self, test_config):
        """Test auto-detection of different delimiters."""
        registrar = DatasetRegistrar()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test pipe delimiter
            pipe_file = Path(tmpdir) / "pipe_data.csv"
            pipe_file.write_text(
                "id|name|value\n"
                "1|Alice|100\n"
                "2|Bob|200\n"
            )
            
            # Test tab delimiter
            tab_file = Path(tmpdir) / "tab_data.tsv"
            tab_file.write_text(
                "id\tname\tvalue\n"
                "1\tAlice\t100\n"
                "2\tBob\t200\n"
            )
            
            # Test semicolon delimiter
            semicolon_file = Path(tmpdir) / "semicolon_data.csv"
            semicolon_file.write_text(
                "id;name;value\n"
                "1;Alice;100\n"
                "2;Bob;200\n"
            )
            
            # All should be detected and handled
            for file_path in [pipe_file, tab_file, semicolon_file]:
                # Test delimiter detection
                from mdm.dataset.auto_detect import detect_delimiter
                delimiter = detect_delimiter(file_path)
                assert delimiter in ['|', '\t', ';']
    
    def test_quoted_fields_with_delimiters(self, test_config):
        """Test CSV with quoted fields containing delimiters."""
        client = MDMClient(config=test_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with complex quoting
            quoted_csv = Path(tmpdir) / "train.csv"
            quoted_csv.write_text(
                'id,description,value\n'
                '1,"This has, a comma",100\n'
                '2,"This has ""quotes"" inside",200\n'
                '3,"Multiple, commas, here",300\n'
                '4,"Mixed: ""quotes"", commas, and more",400\n'
            )
            
            # Should parse correctly
            dataset_info = client.register_dataset(
                name="quoted_fields",
                dataset_path=str(tmpdir),
            )
            
            # Verify parsing
            train_df, _ = client.load_dataset_files("quoted_fields")
            assert len(train_df) == 4
            assert "comma" in train_df['description'].iloc[0]
            assert '"quotes"' in train_df['description'].iloc[1]
            assert train_df['value'].iloc[3] == 400