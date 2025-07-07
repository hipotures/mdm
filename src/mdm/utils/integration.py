"""Integration utilities for ML frameworks."""

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLFrameworkAdapter:
    """Adapter for various ML frameworks."""

    def __init__(self, framework: str = 'auto'):
        """Initialize adapter.
        
        Args:
            framework: ML framework ('sklearn', 'pytorch', 'tensorflow', 'auto')
        """
        self.framework = framework
        self._detect_framework()

    def _detect_framework(self) -> None:
        """Auto-detect available ML frameworks."""
        if self.framework != 'auto':
            return

        try:
            import sklearn
            self.framework = 'sklearn'
        except ImportError:
            pass

        try:
            import torch
            self.framework = 'pytorch'
        except ImportError:
            pass

        try:
            import tensorflow
            self.framework = 'tensorflow'
        except ImportError:
            pass

        if self.framework == 'auto':
            logger.warning("No ML framework detected, using numpy arrays")
            self.framework = 'numpy'

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        feature_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Prepare data for ML framework.
        
        Args:
            train_df: Training DataFrame
            test_df: Optional test DataFrame
            target_column: Target column name
            id_columns: ID columns to exclude
            feature_columns: Specific feature columns to use
            
        Returns:
            Dictionary with prepared data
        """
        # Determine columns
        id_cols = id_columns or []

        if feature_columns:
            feature_cols = feature_columns
        else:
            # All columns except target and IDs
            exclude_cols = id_cols.copy()
            if target_column:
                exclude_cols.append(target_column)
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        # Prepare training data
        X_train = train_df[feature_cols]
        y_train = train_df[target_column] if target_column else None

        # Prepare test data
        X_test = None
        if test_df is not None:
            X_test = test_df[feature_cols]

        # Convert based on framework
        if self.framework == 'sklearn':
            return self._prepare_sklearn(X_train, y_train, X_test)
        if self.framework == 'pytorch':
            return self._prepare_pytorch(X_train, y_train, X_test)
        if self.framework == 'tensorflow':
            return self._prepare_tensorflow(X_train, y_train, X_test)
        return self._prepare_numpy(X_train, y_train, X_test)

    def _prepare_sklearn(self, X_train, y_train, X_test):
        """Prepare data for scikit-learn."""
        result = {
            'X_train': X_train.values,
            'y_train': y_train.values if y_train is not None else None,
            'feature_names': X_train.columns.tolist(),
            'framework': 'sklearn'
        }

        if X_test is not None:
            result['X_test'] = X_test.values

        return result

    def _prepare_pytorch(self, X_train, y_train, X_test):
        """Prepare data for PyTorch."""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)

            if y_train is not None:
                y_train_tensor = torch.FloatTensor(y_train.values)
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            else:
                train_dataset = TensorDataset(X_train_tensor)

            result = {
                'train_dataset': train_dataset,
                'train_loader': DataLoader(train_dataset, batch_size=32, shuffle=True),
                'X_train_tensor': X_train_tensor,
                'y_train_tensor': y_train_tensor if y_train is not None else None,
                'feature_names': X_train.columns.tolist(),
                'framework': 'pytorch'
            }

            if X_test is not None:
                X_test_tensor = torch.FloatTensor(X_test.values)
                test_dataset = TensorDataset(X_test_tensor)
                result['test_dataset'] = test_dataset
                result['test_loader'] = DataLoader(test_dataset, batch_size=32, shuffle=False)
                result['X_test_tensor'] = X_test_tensor

            return result

        except ImportError:
            logger.warning("PyTorch not available, falling back to numpy")
            return self._prepare_numpy(X_train, y_train, X_test)

    def _prepare_tensorflow(self, X_train, y_train, X_test):
        """Prepare data for TensorFlow."""
        try:
            import tensorflow as tf

            # Create dataset
            if y_train is not None:
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices(X_train.values)

            train_dataset = train_dataset.batch(32).shuffle(buffer_size=1000)

            result = {
                'train_dataset': train_dataset,
                'X_train': X_train.values,
                'y_train': y_train.values if y_train is not None else None,
                'feature_names': X_train.columns.tolist(),
                'framework': 'tensorflow'
            }

            if X_test is not None:
                test_dataset = tf.data.Dataset.from_tensor_slices(X_test.values)
                test_dataset = test_dataset.batch(32)
                result['test_dataset'] = test_dataset
                result['X_test'] = X_test.values

            return result

        except ImportError:
            logger.warning("TensorFlow not available, falling back to numpy")
            return self._prepare_numpy(X_train, y_train, X_test)

    def _prepare_numpy(self, X_train, y_train, X_test):
        """Prepare data as numpy arrays."""
        result = {
            'X_train': X_train.values,
            'y_train': y_train.values if y_train is not None else None,
            'feature_names': X_train.columns.tolist(),
            'framework': 'numpy'
        }

        if X_test is not None:
            result['X_test'] = X_test.values

        return result


class DatasetIterator:
    """Iterator for large datasets."""

    def __init__(
        self,
        dataset_manager,
        dataset_name: str,
        table_name: str = 'train',
        batch_size: int = 1000,
        columns: Optional[list[str]] = None,
    ):
        """Initialize dataset iterator.
        
        Args:
            dataset_manager: DatasetManager instance
            dataset_name: Dataset name
            table_name: Table to iterate over
            batch_size: Batch size for iteration
            columns: Specific columns to load
        """
        self.dataset_manager = dataset_manager
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.batch_size = batch_size
        self.columns = columns

        # Get dataset info
        self.dataset_info = dataset_manager.get_dataset(dataset_name)
        if not self.dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Get table info
        self.table_full_name = self.dataset_info.tables.get(table_name)
        if not self.table_full_name:
            raise ValueError(f"Table '{table_name}' not found in dataset")

        # Get connection
        self.conn = dataset_manager.get_dataset_connection(dataset_name)

        # Get total rows
        query = f"SELECT COUNT(*) FROM {self.table_full_name}"
        self.total_rows = self.conn.execute(query).fetchone()[0]
        self.current_offset = 0

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self) -> pd.DataFrame:
        """Get next batch."""
        if self.current_offset >= self.total_rows:
            raise StopIteration

        # Build query
        columns_str = '*' if not self.columns else ', '.join(self.columns)
        query = f"""
            SELECT {columns_str}
            FROM {self.table_full_name}
            LIMIT {self.batch_size}
            OFFSET {self.current_offset}
        """

        # Execute query
        batch_df = self.conn.execute(query).fetch_df()

        # Update offset
        self.current_offset += self.batch_size

        return batch_df

    def __len__(self) -> int:
        """Get number of batches."""
        return (self.total_rows + self.batch_size - 1) // self.batch_size

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self.current_offset = 0


class SubmissionCreator:
    """Create competition submissions."""

    def __init__(self, dataset_manager):
        """Initialize submission creator.
        
        Args:
            dataset_manager: DatasetManager instance
        """
        self.dataset_manager = dataset_manager

    def create_submission(
        self,
        dataset_name: str,
        predictions: Union[np.ndarray, pd.Series, pd.DataFrame, list],
        output_path: Optional[str] = None,
        submission_template: Optional[str] = None,
    ) -> str:
        """Create submission file.
        
        Args:
            dataset_name: Dataset name
            predictions: Model predictions
            output_path: Output file path
            submission_template: Template name ('submission' or custom)
            
        Returns:
            Path to created submission file
        """
        # Get dataset info
        dataset_info = self.dataset_manager.get_dataset(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Get submission template
        template_name = submission_template or 'submission'
        template_table = dataset_info.tables.get(template_name)

        if template_table:
            # Load template
            backend = self.dataset_manager.get_backend(dataset_name)
            template_df = backend.read_table(template_table)
        else:
            # Create from test data
            test_table = dataset_info.tables.get('test')
            if not test_table:
                raise ValueError("No test table or submission template found")

            backend = self.dataset_manager.get_backend(dataset_name)
            test_df = backend.read_table(test_table)

            # Create template with ID columns
            id_cols = dataset_info.id_columns or ['id']
            available_id_cols = [col for col in id_cols if col in test_df.columns]

            if available_id_cols:
                template_df = test_df[available_id_cols].copy()
            else:
                # Use index as ID
                template_df = pd.DataFrame({'id': range(len(test_df))})

        # Add predictions
        if isinstance(predictions, pd.DataFrame):
            # Merge predictions
            for col in predictions.columns:
                template_df[col] = predictions[col].values
        else:
            # Single column predictions
            if isinstance(predictions, pd.Series):
                pred_values = predictions.values
                pred_name = predictions.name or dataset_info.target_column or 'prediction'
            else:
                pred_values = predictions
                pred_name = dataset_info.target_column or 'prediction'

            template_df[pred_name] = pred_values

        # Save submission
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{dataset_name}_submission_{timestamp}.csv"

        template_df.to_csv(output_path, index=False)
        logger.info(f"Created submission file: {output_path}")

        return output_path
