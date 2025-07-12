"""Comprehensive tests for dataset update functionality."""

import pytest
from unittest.mock import Mock, patch
from typer.testing import CliRunner

from mdm.cli.dataset import dataset_app

runner = CliRunner()


class TestDatasetUpdateBehavior:
    """Test dataset update command behavior and edge cases."""

    def test_update_no_changes_returns_success(self):
        """Verify exit code 0 when no updates specified."""
        result = runner.invoke(dataset_app, ["update", "test_dataset"])
        
        assert result.exit_code == 0
        assert "No updates specified" in result.stdout

    def test_update_with_empty_string_description(self):
        """Test update with empty string description."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", ""
            ])
            
            assert result.exit_code == 0
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                description="",
                target=None,
                problem_type=None,
                id_columns=None,
                tags=None
            )

    def test_update_with_malformed_id_columns(self):
        """Test update with various malformed id_columns formats."""
        test_cases = [
            ("", True, None),  # Empty string - no update
            (",", False, None),  # Just comma - should error
            (",,", False, None),  # Multiple commas - should error
            ("col1,", True, ["col1"]),  # Trailing comma
            (",col1", True, ["col1"]),  # Leading comma
            ("col1,,col2", True, ["col1", "col2"]),  # Double comma
            ("  col1  ,  col2  ", True, ["col1", "col2"]),  # Spaces
            ("   ", True, None),  # Only spaces - no columns to add
        ]
        
        for input_str, should_succeed, expected_list in test_cases:
            with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
                mock_update_op = Mock()
                mock_update_op_class.return_value = mock_update_op
                
                result = runner.invoke(dataset_app, [
                    "update", "test_dataset",
                    "--id-columns", input_str
                ])
                
                if should_succeed:
                    assert result.exit_code == 0, f"Failed for input: '{input_str}', output: {result.stdout}"
                    if expected_list:  # Only called if there's actual content
                        mock_update_op.execute.assert_called_with(
                            "test_dataset",
                            description=None,
                            target=None,
                            problem_type=None,
                            id_columns=expected_list,
                            tags=None
                        )
                    else:
                        # Empty string case - no updates
                        assert "No updates specified" in result.stdout
                else:
                    assert result.exit_code == 1
                    assert "Invalid id_columns format" in result.stdout

    def test_update_with_invalid_problem_type(self):
        """Test update with invalid problem type (should fail at CLI level)."""
        result = runner.invoke(dataset_app, [
            "update", "test_dataset",
            "--problem-type", "invalid_type"
        ])
        
        assert result.exit_code == 1
        assert "Invalid problem type 'invalid_type'" in result.stdout
        assert "Valid options are:" in result.stdout
        assert "regression" in result.stdout

    def test_update_with_valid_problem_types(self):
        """Test update with all valid problem types."""
        valid_types = ["binary_classification", "multiclass_classification", "regression", "time_series", "clustering"]
        
        for problem_type in valid_types:
            with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
                mock_update_op = Mock()
                mock_update_op_class.return_value = mock_update_op
                
                result = runner.invoke(dataset_app, [
                    "update", "test_dataset",
                    "--problem-type", problem_type
                ])
                
                assert result.exit_code == 0
                mock_update_op.execute.assert_called_with(
                    "test_dataset",
                    description=None,
                    target=None,
                    problem_type=problem_type,
                    id_columns=None,
                    tags=None
                )

    def test_update_multiple_fields_simultaneously(self):
        """Test updating multiple fields at once."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", "New description",
                "--target", "new_target",
                "--problem-type", "regression",
                "--id-columns", "id1,id2"
            ])
            
            assert result.exit_code == 0
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                description="New description",
                target="new_target",
                problem_type="regression",
                id_columns=["id1", "id2"],
                tags=None
            )

    def test_update_handles_dataset_error(self):
        """Test proper error handling when UpdateOperation raises DatasetError."""
        from mdm.core.exceptions import DatasetError
        
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op.execute.side_effect = DatasetError("Dataset not found")
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "nonexistent_dataset",
                "--description", "New description"
            ])
            
            assert result.exit_code == 1
            assert "Error: Dataset not found" in result.stdout

    def test_update_handles_generic_error(self):
        """Test proper error handling for generic exceptions (no info leakage)."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op.execute.side_effect = Exception("Internal database connection failed at host:port with credentials")
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", "New description"
            ])
            
            assert result.exit_code == 1
            assert "Failed to update dataset" in result.stdout
            assert "Check logs for details" in result.stdout
            # Ensure sensitive info is not leaked
            assert "host:port" not in result.stdout
            assert "credentials" not in result.stdout

    def test_update_success_output_format(self):
        """Test the output format when update succeeds."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", "Updated description",
                "--target", "price"
            ])
            
            assert result.exit_code == 0
            assert "✓" in result.stdout
            assert "Dataset 'test_dataset' updated successfully" in result.stdout
            assert "Updated fields:" in result.stdout
            assert "description: Updated description" in result.stdout
            assert "target_column: price" in result.stdout


class TestDatasetUpdateInputValidation:
    """Test input validation for dataset update."""

    def test_update_special_characters_in_description(self):
        """Test update with special characters in description."""
        special_chars = "Test with <script>alert('xss')</script> & SQL: DROP TABLE;"
        
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", special_chars
            ])
            
            assert result.exit_code == 0
            # Should pass through without modification at CLI level
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                description=special_chars,
                target=None,
                problem_type=None,
                id_columns=None,
                tags=None
            )

    def test_update_very_long_description(self):
        """Test update with very long description."""
        long_description = "A" * 10000
        
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", long_description
            ])
            
            assert result.exit_code == 0
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                description=long_description,
                target=None,
                problem_type=None,
                id_columns=None,
                tags=None
            )

    def test_update_unicode_in_fields(self):
        """Test update with unicode characters."""
        unicode_text = "测试数据集 🚀 Тест"
        
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", unicode_text
            ])
            
            assert result.exit_code == 0
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                description=unicode_text,
                target=None,
                problem_type=None,
                id_columns=None,
                tags=None
            )

    def test_update_whitespace_only_fields(self):
        """Test update with whitespace-only values."""
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(dataset_app, [
                "update", "test_dataset",
                "--description", "   ",
                "--target", "\t\n"
            ])
            
            assert result.exit_code == 0
            # Whitespace is preserved as-is
            mock_update_op.execute.assert_called_with(
                "test_dataset",
                description="   ",
                target="\t\n",
                problem_type=None,
                id_columns=None,
                tags=None
            )


class TestDatasetUpdateEdgeCases:
    """Test edge cases for dataset update."""

    def test_update_with_none_values(self):
        """Test that None values are properly handled."""
        # This test verifies the CLI logic that filters out None values
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            # When no options are provided, all values are None
            result = runner.invoke(dataset_app, ["update", "test_dataset"])
            
            assert result.exit_code == 0
            assert "No updates specified" in result.stdout
            # execute should not be called
            mock_update_op.execute.assert_not_called()

    def test_update_dataset_name_with_special_chars(self):
        """Test update with special characters in dataset name."""
        dataset_names = [
            "test-dataset",
            "test_dataset_2",
            "test.dataset",
            "TEST_DATASET",
        ]
        
        for name in dataset_names:
            with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
                mock_update_op = Mock()
                mock_update_op_class.return_value = mock_update_op
                
                result = runner.invoke(dataset_app, [
                    "update", name,
                    "--description", "Test"
                ])
                
                assert result.exit_code == 0
                mock_update_op.execute.assert_called_with(
                    name,
                    description="Test",
                    target=None,
                    problem_type=None,
                    id_columns=None,
                    tags=None
                )

    def test_update_with_typer_context_preserved(self):
        """Test that typer context is properly preserved during update."""
        # This ensures that any typer-specific context (like color settings)
        # is preserved throughout the command execution
        with patch('mdm.dataset.operations.UpdateOperation') as mock_update_op_class:
            mock_update_op = Mock()
            mock_update_op_class.return_value = mock_update_op
            
            result = runner.invoke(
                dataset_app,
                ["update", "test_dataset", "--description", "Test"],
                env={"NO_COLOR": "1"}  # Disable colors
            )
            
            assert result.exit_code == 0
            # Check that ANSI codes are not present
            assert "\033[" not in result.stdout