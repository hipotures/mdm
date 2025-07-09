"""Unit tests for auto-detection utilities."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import pandas as pd

from mdm.dataset.auto_detect import (
    detect_delimiter,
    detect_id_columns,
    detect_kaggle_structure,
    discover_data_files,
    extract_target_from_sample_submission,
    infer_problem_type,
    validate_kaggle_submission_format
)
from mdm.models.enums import ProblemType


class TestAutoDetect:
    """Test cases for auto-detection utilities."""

    def test_detect_delimiter_comma(self):
        """Test comma delimiter detection."""
        csv_content = "id,name,value\n1,test,100\n2,test2,200"
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            result = detect_delimiter(Path("test.csv"))
        
        assert result == ","

    def test_detect_delimiter_tab(self):
        """Test tab delimiter detection."""
        tsv_content = "id\tname\tvalue\n1\ttest\t100\n2\ttest2\t200"
        
        with patch('builtins.open', mock_open(read_data=tsv_content)):
            result = detect_delimiter(Path("test.tsv"))
        
        assert result == "\t"

    def test_detect_delimiter_semicolon(self):
        """Test semicolon delimiter detection."""
        csv_content = "id;name;value\n1;test;100\n2;test2;200"
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            result = detect_delimiter(Path("test.csv"))
        
        assert result == ";"

    def test_detect_delimiter_pipe(self):
        """Test pipe delimiter detection."""
        csv_content = "id|name|value\n1|test|100\n2|test2|200"
        
        with patch('builtins.open', mock_open(read_data=csv_content)):
            result = detect_delimiter(Path("test.csv"))
        
        assert result == "|"

    def test_detect_id_columns_single(self):
        """Test detection of single ID column."""
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'name': ['a', 'b', 'c', 'd', 'e'],
            'value': [10, 20, 30, 40, 50]
        })
        
        result = detect_id_columns(df)
        assert result == ['user_id']

    def test_detect_id_columns_multiple(self):
        """Test detection of multiple ID columns."""
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'session_id': ['s1', 's2', 's3', 's4', 's5'],
            'value': [10, 10, 20, 20, 30]  # Duplicates
        })
        
        result = detect_id_columns(df)
        assert set(result) == {'user_id', 'session_id'}

    def test_detect_id_columns_index_column(self):
        """Test detection of index column."""
        df = pd.DataFrame({
            'index': [0, 1, 2, 3, 4],
            'value': [10, 20, 30, 40, 50]
        })
        
        result = detect_id_columns(df)
        assert result == ['index']

    def test_detect_id_columns_no_suitable(self):
        """Test when no suitable ID columns found."""
        df = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],  # All same
            'feature2': [2, 2, 2, 2, 2]   # All same
        })
        
        result = detect_id_columns(df)
        assert result == []

    def test_detect_kaggle_structure_complete(self):
        """Test detection of complete Kaggle structure."""
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.is_dir', return_value=True):
                # Mock file existence checks
                mock_exists.side_effect = lambda: True
                
                path = Path("/kaggle/dataset")
                result = detect_kaggle_structure(path)
        
        assert result['is_kaggle'] is True
        assert result['train_file'].name == "train.csv"
        assert result['test_file'].name == "test.csv"
        assert result['submission_file'].name == "sample_submission.csv"

    def test_detect_kaggle_structure_partial(self):
        """Test detection with only train and test files."""
        def exists_check(self):
            return self.name in ['train.csv', 'test.csv']
        
        with patch('pathlib.Path.exists', exists_check):
            with patch('pathlib.Path.is_dir', return_value=True):
                path = Path("/kaggle/dataset")
                result = detect_kaggle_structure(path)
        
        assert result['is_kaggle'] is True
        assert result['submission_file'] is None

    def test_detect_kaggle_structure_not_kaggle(self):
        """Test non-Kaggle structure."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.is_dir', return_value=True):
                path = Path("/data/custom")
                result = detect_kaggle_structure(path)
        
        assert result['is_kaggle'] is False

    def test_discover_data_files_directory(self):
        """Test discovering files in directory."""
        mock_files = [
            Path("data.csv"),
            Path("data.xlsx"),
            Path("data.parquet"),
            Path("readme.txt"),  # Should be ignored
            Path("subdir/more_data.csv")
        ]
        
        with patch('pathlib.Path.is_dir', return_value=True):
            with patch('pathlib.Path.rglob') as mock_rglob:
                mock_rglob.return_value = mock_files
                
                result = discover_data_files(Path("/data"))
        
        assert len(result) == 4
        assert any(f.name == "data.csv" for f in result)
        assert any(f.name == "data.xlsx" for f in result)
        assert any(f.name == "data.parquet" for f in result)
        assert not any(f.name == "readme.txt" for f in result)

    def test_discover_data_files_single_file(self):
        """Test discovering single file."""
        file_path = Path("/data/dataset.csv")
        
        with patch('pathlib.Path.is_dir', return_value=False):
            with patch('pathlib.Path.exists', return_value=True):
                result = discover_data_files(file_path)
        
        assert len(result) == 1
        assert result[0] == file_path

    def test_extract_target_from_sample_submission(self):
        """Test extracting target column from submission file."""
        submission_df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [0.1, 0.2, 0.3]
        })
        
        with patch('pandas.read_csv', return_value=submission_df):
            result = extract_target_from_sample_submission(Path("submission.csv"))
        
        assert result == 'target'

    def test_extract_target_multiple_candidates(self):
        """Test target extraction with multiple non-ID columns."""
        submission_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'prediction': [0.1, 0.2, 0.3],
            'confidence': [0.9, 0.8, 0.7]
        })
        
        with patch('pandas.read_csv', return_value=submission_df):
            # Should return first non-ID column
            result = extract_target_from_sample_submission(Path("submission.csv"))
        
        assert result in ['prediction', 'confidence']

    def test_infer_problem_type_binary_classification(self):
        """Test inferring binary classification."""
        target_values = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
        
        result = infer_problem_type(target_values)
        assert result == ProblemType.CLASSIFICATION

    def test_infer_problem_type_multiclass(self):
        """Test inferring multiclass classification."""
        target_values = pd.Series([0, 1, 2, 1, 2, 0, 3, 2, 1])
        
        result = infer_problem_type(target_values)
        assert result == ProblemType.CLASSIFICATION

    def test_infer_problem_type_regression(self):
        """Test inferring regression."""
        target_values = pd.Series([1.5, 2.7, 3.14, 4.2, 5.9, 6.1])
        
        result = infer_problem_type(target_values)
        assert result == ProblemType.REGRESSION

    def test_infer_problem_type_string_classification(self):
        """Test inferring classification from string targets."""
        target_values = pd.Series(['cat', 'dog', 'cat', 'bird', 'dog'])
        
        result = infer_problem_type(target_values)
        assert result == ProblemType.CLASSIFICATION

    def test_validate_kaggle_submission_format_valid(self):
        """Test validating correct submission format."""
        train_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': [10, 20, 30],
            'target': [0, 1, 0]
        })
        
        test_df = pd.DataFrame({
            'id': [4, 5, 6],
            'feature': [40, 50, 60]
        })
        
        submission_df = pd.DataFrame({
            'id': [4, 5, 6],
            'target': [0, 0, 1]
        })
        
        # Should not raise any exception
        validate_kaggle_submission_format(train_df, test_df, submission_df)

    def test_validate_kaggle_submission_format_missing_ids(self):
        """Test validation fails with missing test IDs."""
        train_df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        test_df = pd.DataFrame({
            'id': [4, 5, 6]
        })
        
        submission_df = pd.DataFrame({
            'id': [4, 5],  # Missing ID 6
            'target': [0, 0]
        })
        
        with pytest.raises(ValueError, match="Submission IDs don't match"):
            validate_kaggle_submission_format(train_df, test_df, submission_df)

    def test_validate_kaggle_submission_format_no_target(self):
        """Test validation with no target in submission."""
        train_df = pd.DataFrame({
            'id': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        test_df = pd.DataFrame({
            'id': [4, 5, 6]
        })
        
        submission_df = pd.DataFrame({
            'id': [4, 5, 6]
            # No target column
        })
        
        with pytest.raises(ValueError, match="No prediction column found"):
            validate_kaggle_submission_format(train_df, test_df, submission_df)