"""Advanced dataset service for low-level operations."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from mdm.config import get_config
from mdm.dataset.manager import DatasetManager
from mdm.dataset.registrar import DatasetRegistrar
from mdm.features.engine import FeatureEngine

logger = logging.getLogger(__name__)


class DatasetService:
    """Advanced service for low-level dataset operations."""

    def __init__(self, manager: Optional[DatasetManager] = None):
        """Initialize dataset service.

        Args:
            manager: Optional DatasetManager instance
        """
        self.config = get_config()
        self.manager = manager or DatasetManager()
        self.registrar = DatasetRegistrar(self.manager)
        self.feature_engine = FeatureEngine()

    def register_dataset_auto(
        self,
        name: str,
        path: str,
        target_column: Optional[str] = None,
        id_column: Optional[str] = None,
        competition_name: Optional[str] = None,
        description: Optional[str] = None,
        force_update: bool = False,
    ) -> dict[str, Any]:
        """Auto-register dataset with intelligent detection.

        Args:
            name: Dataset name
            path: Path to dataset directory
            target_column: Target column name
            id_column: ID column name
            competition_name: Competition name (for Kaggle datasets)
            description: Dataset description
            force_update: Whether to force update existing dataset

        Returns:
            Registration result dictionary
        """
        dataset_path = Path(path)

        # Add competition tag if specified
        tags = []
        if competition_name:
            tags.append(f"competition:{competition_name}")

        # Register dataset
        dataset_info = self.registrar.register(
            name=name,
            path=dataset_path,
            auto_detect=True,
            target_column=target_column,
            id_columns=[id_column] if id_column else None,
            description=description,
            tags=tags,
            force=force_update,
        )

        return {
            "success": True,
            "dataset_info": dataset_info,
            "tables": dataset_info.tables,
            "feature_tables": dataset_info.feature_tables if hasattr(dataset_info, 'feature_tables') else {},
        }

    def register_dataset(
        self,
        name: str,
        train_path: str,
        test_path: Optional[str] = None,
        validation_path: Optional[str] = None,
        submission_path: Optional[str] = None,
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        **kwargs
    ) -> dict[str, Any]:
        """Register dataset with specific file paths.

        Args:
            name: Dataset name
            train_path: Path to training data
            test_path: Path to test data
            validation_path: Path to validation data
            submission_path: Path to submission template
            target_column: Target column name
            id_columns: List of ID columns
            **kwargs: Additional options

        Returns:
            Registration result dictionary
        """
        # Create temporary directory structure
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy files to temporary structure
            import shutil
            train_dest = temp_path / "train.csv"
            shutil.copy2(train_path, train_dest)

            if test_path:
                test_dest = temp_path / "test.csv"
                shutil.copy2(test_path, test_dest)

            if validation_path:
                val_dest = temp_path / "validation.csv"
                shutil.copy2(validation_path, val_dest)

            if submission_path:
                sub_dest = temp_path / "sample_submission.csv"
                shutil.copy2(submission_path, sub_dest)

            # Register from temporary directory
            dataset_info = self.registrar.register(
                name=name,
                path=temp_path,
                auto_detect=True,
                target_column=target_column,
                id_columns=id_columns,
                **kwargs
            )

        return {
            "success": True,
            "dataset_info": dataset_info,
            "tables": dataset_info.tables,
        }

    def generate_features(
        self,
        dataset_name: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Generate features for a dataset.

        Args:
            dataset_name: Name of the dataset
            force: Whether to regenerate existing features

        Returns:
            Feature generation results
        """
        dataset_info = self.manager.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Check if features already exist
        if hasattr(dataset_info, 'feature_tables') and dataset_info.feature_tables and not force:
            return {
                "success": True,
                "message": "Features already exist",
                "feature_tables": dataset_info.feature_tables,
            }

        # Get backend
        backend = self.manager.get_backend(dataset_name)

        # Generate features
        feature_info = self.feature_engine.generate_features(
            dataset_name=dataset_name,
            backend=backend,
            tables=dataset_info.tables,
            target_column=dataset_info.target_column,
            id_columns=dataset_info.id_columns,
        )

        # Update dataset info with feature tables
        feature_tables = {
            table_type: info['feature_table']
            for table_type, info in feature_info.items()
        }

        self.manager.update_dataset(dataset_name, {'feature_tables': feature_tables})

        return {
            "success": True,
            "feature_info": feature_info,
            "feature_tables": feature_tables,
        }

    def create_submission(
        self,
        dataset_name: str,
        predictions: Union[pd.Series, pd.DataFrame, list],
        output_path: Optional[str] = None,
    ) -> str:
        """Create submission file for a dataset.

        Args:
            dataset_name: Dataset name
            predictions: Predictions (Series, DataFrame, or list)
            output_path: Output file path (default: submission.csv)

        Returns:
            Path to created submission file
        """
        dataset_info = self.manager.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Get submission template if exists
        submission_table = dataset_info.tables.get('submission')
        if submission_table:
            backend = self.manager.get_backend(dataset_name)
            template = backend.read_table(submission_table)
        else:
            # Create basic template
            test_table = dataset_info.tables.get('test')
            if not test_table:
                raise ValueError(f"Dataset '{dataset_name}' has no test table")

            backend = self.manager.get_backend(dataset_name)
            test_df = backend.read_table(test_table)

            # Get ID columns
            id_cols = dataset_info.id_columns or ['id']
            available_id_cols = [col for col in id_cols if col in test_df.columns]

            if not available_id_cols:
                # Use index as ID
                template = pd.DataFrame({'id': range(len(test_df))})
            else:
                template = test_df[available_id_cols].copy()

        # Add predictions
        if isinstance(predictions, pd.Series):
            pred_col = predictions.name or dataset_info.target_column or 'prediction'
            template[pred_col] = predictions.values
        elif isinstance(predictions, pd.DataFrame):
            # Merge predictions
            for col in predictions.columns:
                template[col] = predictions[col].values
        else:
            # List or array
            pred_col = dataset_info.target_column or 'prediction'
            template[pred_col] = predictions

        # Save submission
        if output_path is None:
            output_path = f"{dataset_name}_submission.csv"

        template.to_csv(output_path, index=False)
        logger.info(f"Created submission file: {output_path}")

        return output_path

    def split_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        stratify: bool = True,
        random_state: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Split dataset into train/test/validation sets.

        Args:
            dataset_name: Dataset name
            test_size: Test set size (fraction)
            validation_size: Validation set size (fraction)
            stratify: Whether to stratify split by target
            random_state: Random seed

        Returns:
            Dictionary with train, test, and optionally validation DataFrames
        """
        from sklearn.model_selection import train_test_split

        dataset_info = self.manager.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Load full dataset
        backend = self.manager.get_backend(dataset_name)
        train_table = dataset_info.tables.get('train')
        if not train_table:
            raise ValueError(f"Dataset '{dataset_name}' has no train table")

        df = backend.read_table(train_table)

        # Prepare stratification
        stratify_col = None
        if stratify and dataset_info.target_column and dataset_info.target_column in df.columns:
            stratify_col = df[dataset_info.target_column]

        # Split train/test
        if validation_size > 0:
            # Three-way split
            train_val_size = 1.0 - test_size
            val_size_adjusted = validation_size / train_val_size

            # First split: train+val vs test
            X_temp, X_test = train_test_split(
                df,
                test_size=test_size,
                stratify=stratify_col,
                random_state=random_state
            )

            # Second split: train vs val
            stratify_temp = None
            if stratify_col is not None:
                stratify_temp = X_temp[dataset_info.target_column]

            X_train, X_val = train_test_split(
                X_temp,
                test_size=val_size_adjusted,
                stratify=stratify_temp,
                random_state=random_state
            )

            return {
                'train': X_train,
                'validation': X_val,
                'test': X_test,
            }
        # Two-way split
        X_train, X_test = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )

        return {
            'train': X_train,
            'test': X_test,
        }

    def analyze_dataset(self, dataset_name: str) -> dict[str, Any]:
        """Perform comprehensive dataset analysis.

        Args:
            dataset_name: Dataset name

        Returns:
            Analysis results dictionary
        """
        from mdm.dataset.operations import StatsOperation

        dataset_info = self.manager.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Get statistics
        stats_op = StatsOperation()
        stats = stats_op.execute(dataset_name, full=True)

        # Additional analysis
        backend = self.manager.get_backend(dataset_name)
        train_table = dataset_info.tables.get('train')

        analysis = {
            'basic_info': {
                'name': dataset_info.name,
                'problem_type': dataset_info.problem_type,
                'target_column': dataset_info.target_column,
                'id_columns': dataset_info.id_columns,
                'tables': list(dataset_info.tables.keys()),
            },
            'statistics': stats,
        }

        if train_table:
            df_sample = backend.read_table(train_table, limit=1000)

            # Data quality metrics
            quality_metrics = {
                'missing_values': df_sample.isnull().sum().to_dict(),
                'duplicate_rows': df_sample.duplicated().sum(),
                'unique_values': {col: df_sample[col].nunique() for col in df_sample.columns},
            }

            analysis['data_quality'] = quality_metrics

            # Target distribution if available
            if dataset_info.target_column and dataset_info.target_column in df_sample.columns:
                target_dist = df_sample[dataset_info.target_column].value_counts().to_dict()
                analysis['target_distribution'] = target_dist

        return analysis
