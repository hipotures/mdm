"""Unit tests for DatasetService."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import pandas as pd
import tempfile
import shutil

from mdm.services.dataset_service import DatasetService
from mdm.models.dataset import DatasetInfo
from mdm.models.enums import ProblemType


class TestDatasetService:
    """Test cases for DatasetService."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock DatasetManager."""
        manager = Mock()
        manager.get_dataset.return_value = None
        manager.update_dataset.return_value = None
        return manager

    @pytest.fixture
    def mock_registrar(self):
        """Create mock DatasetRegistrar."""
        registrar = Mock()
        return registrar

    @pytest.fixture
    def mock_feature_engine(self):
        """Create mock FeatureEngine."""
        engine = Mock()
        return engine

    @pytest.fixture
    def service(self, mock_manager, mock_registrar, mock_feature_engine):
        """Create DatasetService instance with mocked dependencies."""
        with patch('mdm.services.dataset_service.DatasetRegistrar', return_value=mock_registrar):
            with patch('mdm.services.dataset_service.FeatureEngine', return_value=mock_feature_engine):
                service = DatasetService(manager=mock_manager)
                service.registrar = mock_registrar
                service.feature_engine = mock_feature_engine
                return service

    @pytest.fixture
    def sample_dataset_info(self):
        """Create sample DatasetInfo."""
        return DatasetInfo(
            name="test_dataset",
            problem_type="binary_classification",
            target_column="target",
            id_columns=["id"],
            tables={
                "train": "train_table",
                "test": "test_table",
                "submission": "submission_table"
            },
            shape=(1000, 10),
            description="Test dataset",
            tags=["test", "sample"],
            database={"backend": "sqlite"}
        )

    def test_register_dataset_auto_success(self, service, mock_registrar):
        """Test successful auto-registration of dataset."""
        # Arrange
        expected_info = DatasetInfo(
            name="test_dataset",
            problem_type="binary_classification",
            tables={"train": "train_table"},
            shape=(100, 10),
            database={"backend": "sqlite"}
        )
        mock_registrar.register.return_value = expected_info

        # Act
        result = service.register_dataset_auto(
            name="test_dataset",
            path="/test/path",
            target_column="target",
            id_column="id",
            competition_name="test_competition",
            description="Test description",
            force_update=True
        )

        # Assert
        assert result["success"] is True
        assert result["dataset_info"] == expected_info
        assert result["tables"] == expected_info.tables
        
        mock_registrar.register.assert_called_once_with(
            name="test_dataset",
            path=Path("/test/path"),
            auto_detect=True,
            target_column="target",
            id_columns=["id"],
            description="Test description",
            tags=["competition:test_competition"],
            force=True
        )

    def test_register_dataset_auto_without_competition(self, service, mock_registrar):
        """Test auto-registration without competition tag."""
        # Arrange
        expected_info = DatasetInfo(
            name="test_dataset",
            problem_type="regression",
            tables={"train": "train_table"},
            shape=(100, 10),
            database={"backend": "sqlite"}
        )
        mock_registrar.register.return_value = expected_info

        # Act
        result = service.register_dataset_auto(
            name="test_dataset",
            path="/test/path"
        )

        # Assert
        assert result["success"] is True
        mock_registrar.register.assert_called_once()
        call_args = mock_registrar.register.call_args[1]
        assert call_args["tags"] == []

    @patch('tempfile.TemporaryDirectory')
    @patch('shutil.copy2')
    def test_register_dataset_with_files(self, mock_copy, mock_tempdir, service, mock_registrar):
        """Test registration with specific file paths."""
        # Arrange
        temp_path = "/tmp/test_temp"
        mock_tempdir.return_value.__enter__.return_value = temp_path
        
        expected_info = DatasetInfo(
            name="test_dataset",
            problem_type="binary_classification",
            tables={"train": "train_table"},
            shape=(100, 10),
            database={"backend": "sqlite"}
        )
        mock_registrar.register.return_value = expected_info

        # Act
        result = service.register_dataset(
            name="test_dataset",
            train_path="/path/to/train.csv",
            test_path="/path/to/test.csv",
            validation_path="/path/to/val.csv",
            submission_path="/path/to/submission.csv",
            target_column="target",
            id_columns=["id1", "id2"]
        )

        # Assert
        assert result["success"] is True
        assert result["dataset_info"] == expected_info
        
        # Verify file copying
        assert mock_copy.call_count == 4
        mock_copy.assert_any_call("/path/to/train.csv", Path(temp_path) / "train.csv")
        mock_copy.assert_any_call("/path/to/test.csv", Path(temp_path) / "test.csv")
        mock_copy.assert_any_call("/path/to/val.csv", Path(temp_path) / "validation.csv")
        mock_copy.assert_any_call("/path/to/submission.csv", Path(temp_path) / "sample_submission.csv")

    def test_generate_features_already_exist(self, service, mock_manager, sample_dataset_info):
        """Test feature generation when features already exist."""
        # Arrange
        sample_dataset_info.feature_tables = {"train": "train_features"}
        mock_manager.get_dataset.return_value = sample_dataset_info

        # Act
        result = service.generate_features("test_dataset", force=False)

        # Assert
        assert result["success"] is True
        assert result["message"] == "Features already exist"
        assert result["feature_tables"] == {"train": "train_features"}
        service.feature_engine.generate_features.assert_not_called()

    def test_generate_features_force_regeneration(self, service, mock_manager, mock_feature_engine, sample_dataset_info):
        """Test forced feature regeneration."""
        # Arrange
        sample_dataset_info.feature_tables = {"train": "old_features"}
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        feature_info = {
            "train": {"feature_table": "new_train_features", "features": 20},
            "test": {"feature_table": "new_test_features", "features": 20}
        }
        mock_feature_engine.generate_features.return_value = feature_info

        # Act
        result = service.generate_features("test_dataset", force=True)

        # Assert
        assert result["success"] is True
        assert result["feature_info"] == feature_info
        assert result["feature_tables"] == {
            "train": "new_train_features",
            "test": "new_test_features"
        }
        
        mock_feature_engine.generate_features.assert_called_once_with(
            dataset_name="test_dataset",
            backend=mock_backend,
            tables=sample_dataset_info.tables,
            target_column="target",
            id_columns=["id"]
        )
        
        mock_manager.update_dataset.assert_called_once_with(
            "test_dataset",
            {"feature_tables": {"train": "new_train_features", "test": "new_test_features"}}
        )

    def test_generate_features_dataset_not_found(self, service, mock_manager):
        """Test feature generation with non-existent dataset."""
        # Arrange
        mock_manager.get_dataset.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            service.generate_features("nonexistent")

    def test_create_submission_with_template(self, service, mock_manager, sample_dataset_info):
        """Test submission creation with existing template."""
        # Arrange
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        template_df = pd.DataFrame({
            "id": [1, 2, 3],
            "target": [0, 0, 0]
        })
        mock_backend.read_table.return_value = template_df
        
        predictions = pd.Series([0.1, 0.2, 0.3], name="prediction")

        # Act
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            result = service.create_submission(
                "test_dataset",
                predictions,
                output_path="submission.csv"
            )

        # Assert
        assert result == "submission.csv"
        mock_to_csv.assert_called_once_with("submission.csv", index=False)

    def test_create_submission_without_template(self, service, mock_manager, sample_dataset_info):
        """Test submission creation without template."""
        # Arrange
        sample_dataset_info.tables.pop("submission", None)
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "feature1": [10, 20, 30]
        })
        mock_backend.read_table.return_value = test_df
        
        predictions = [0.1, 0.2, 0.3]

        # Act
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            result = service.create_submission(
                "test_dataset",
                predictions
            )

        # Assert
        assert result == "test_dataset_submission.csv"

    def test_create_submission_with_dataframe_predictions(self, service, mock_manager, sample_dataset_info):
        """Test submission creation with DataFrame predictions."""
        # Arrange
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        template_df = pd.DataFrame({"id": [1, 2, 3]})
        mock_backend.read_table.return_value = template_df
        
        predictions = pd.DataFrame({
            "pred1": [0.1, 0.2, 0.3],
            "pred2": [0.4, 0.5, 0.6]
        })

        # Act
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            result = service.create_submission(
                "test_dataset",
                predictions
            )

        # Assert
        assert result == "test_dataset_submission.csv"

    def test_split_dataset_two_way(self, service, mock_manager, sample_dataset_info):
        """Test two-way dataset split."""
        # Arrange
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        train_df = pd.DataFrame({
            "id": range(100),
            "feature1": range(100),
            "target": [0, 1] * 50
        })
        mock_backend.read_table.return_value = train_df

        # Act
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            mock_split.return_value = (train_df[:80], train_df[80:])
            result = service.split_dataset(
                "test_dataset",
                test_size=0.2,
                validation_size=0.0,
                stratify=True,
                random_state=42
            )

        # Assert
        assert "train" in result
        assert "test" in result
        assert "validation" not in result
        mock_split.assert_called_once()

    def test_split_dataset_three_way(self, service, mock_manager, sample_dataset_info):
        """Test three-way dataset split."""
        # Arrange
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        train_df = pd.DataFrame({
            "id": range(100),
            "feature1": range(100),
            "target": [0, 1] * 50
        })
        mock_backend.read_table.return_value = train_df

        # Act
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            # First split: 80/20
            mock_split.side_effect = [
                (train_df[:80], train_df[80:]),  # train+val vs test
                (train_df[:60], train_df[60:80])  # train vs val
            ]
            result = service.split_dataset(
                "test_dataset",
                test_size=0.2,
                validation_size=0.2,
                stratify=True,
                random_state=42
            )

        # Assert
        assert "train" in result
        assert "validation" in result
        assert "test" in result
        assert mock_split.call_count == 2

    def test_analyze_dataset(self, service, mock_manager, sample_dataset_info):
        """Test comprehensive dataset analysis."""
        # Arrange
        mock_manager.get_dataset.return_value = sample_dataset_info
        mock_backend = Mock()
        mock_manager.get_backend.return_value = mock_backend
        
        sample_df = pd.DataFrame({
            "id": [1, 2, 3, 3],  # Duplicate
            "feature1": [10, None, 30, 40],  # Missing value
            "target": [0, 1, 0, 1]
        })
        mock_backend.read_table.return_value = sample_df
        
        mock_stats_op = Mock()
        mock_stats = {
            "row_count": 100,
            "column_count": 10,
            "missing_values": {"feature1": 1}
        }
        mock_stats_op.execute.return_value = mock_stats

        # Act
        with patch('mdm.dataset.operations.StatsOperation', return_value=mock_stats_op):
            result = service.analyze_dataset("test_dataset")

        # Assert
        assert result["basic_info"]["name"] == "test_dataset"
        assert result["basic_info"]["problem_type"] == ProblemType.CLASSIFICATION
        assert result["basic_info"]["target_column"] == "target"
        assert result["statistics"] == mock_stats
        assert "data_quality" in result
        assert result["data_quality"]["duplicate_rows"] == 1
        assert "target_distribution" in result

    def test_analyze_dataset_not_found(self, service, mock_manager):
        """Test analysis of non-existent dataset."""
        # Arrange
        mock_manager.get_dataset.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            service.analyze_dataset("nonexistent")