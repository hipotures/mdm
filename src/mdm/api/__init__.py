"""MDM API module."""

from typing import Optional, List, Tuple
import pandas as pd

from .mdm_client import MDMClient
from .clients import (
    RegistrationClient,
    QueryClient,
    MLIntegrationClient,
    ExportClient,
    ManagementClient,
)
from mdm.models.dataset import DatasetInfo


# Convenience functions for backward compatibility
def load_dataset(name: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load dataset files.

    Args:
        name: Dataset name

    Returns:
        Tuple of (train_df, test_df)
    """
    client = MDMClient()
    files = client.query.load_dataset_files(name)
    train_df = files.get('train', files.get('data'))
    test_df = files.get('test')
    return train_df, test_df


def list_datasets() -> List[str]:
    """List all dataset names.

    Returns:
        List of dataset names
    """
    client = MDMClient()
    datasets = client.list_datasets()
    return [d.name for d in datasets]


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """Get dataset information.

    Args:
        name: Dataset name

    Returns:
        DatasetInfo object or None
    """
    client = MDMClient()
    return client.get_dataset(name)


__all__ = [
    'MDMClient',
    'RegistrationClient',
    'QueryClient',
    'MLIntegrationClient',
    'ExportClient',
    'ManagementClient',
    'load_dataset',
    'list_datasets',
    'get_dataset_info',
]