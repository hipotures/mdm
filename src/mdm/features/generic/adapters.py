"""Adapters for new feature generators to work with MDM's feature system."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from mdm.features.base import GenericFeatureOperation


class GlobalFeatureAdapter(GenericFeatureOperation):
    """Adapter to make global feature generators work with MDM's column-based system.
    
    This adapter wraps feature generators that work on the entire dataset
    and makes them compatible with the column-by-column processing model.
    """
    
    def __init__(self, feature_generator_class, **kwargs):
        """Initialize the adapter with a feature generator class.
        
        Args:
            feature_generator_class: The class of the global feature generator
            **kwargs: Arguments to pass to the feature generator
        """
        super().__init__(kwargs.get('min_signal_ratio', 0.01))
        self.generator = feature_generator_class(**kwargs)
        self.generator_class_name = feature_generator_class.__name__
        self._generated_features = None
        self._processed_datasets = set()
        
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns this transformer can process.
        
        For global transformers, we return an empty list since they
        don't work on individual columns.
        """
        return []
    
    def _generate_column_features(
        self, df: pd.DataFrame, column: str, **kwargs: Any
    ) -> Dict[str, pd.Series]:
        """Generate features for the entire dataset.
        
        This is called once with the synthetic column name.
        """
        if column != "__global_features__":
            return {}
            
        # Generate features using the wrapped generator
        logger.info(f"Generating global features using {self.generator_class_name}")
        
        try:
            # Call the generator's generate_features method
            features_df, descriptions = self.generator.generate_features(
                df, 
                target_column=kwargs.get('target_column'),
                id_columns=kwargs.get('id_columns', [])
            )
            
            # Convert DataFrame to dict of Series
            features_dict = {}
            for col in features_df.columns:
                features_dict[col] = features_df[col]
                
            # Log the descriptions for debugging
            for feat_name, desc in descriptions.items():
                logger.debug(f"Generated feature '{feat_name}': {desc}")
                
            # Mark this dataset as processed
            dataset_id = f"{id(df)}_{len(df)}_{df.shape[1]}"
            self._processed_datasets.add(dataset_id)
            
            return features_dict
            
        except Exception as e:
            logger.error(f"Error generating {self.generator_class_name} features: {e}")
            return {}


def create_global_adapters():
    """Create adapters for all global feature generators.
    
    Returns:
        List of adapted feature generator instances
    """
    from mdm.features.generic.missing_data import MissingDataFeatures
    from mdm.features.generic.interaction import InteractionFeatures
    from mdm.features.generic.sequential import SequentialFeatures
    from mdm.features.generic.distribution import DistributionFeatures
    from mdm.features.generic.clustering import ClusteringFeatures
    
    adapters = []
    
    # MissingDataFeatures adapter
    adapters.append(GlobalFeatureAdapter(MissingDataFeatures))
    
    # InteractionFeatures adapter
    adapters.append(GlobalFeatureAdapter(InteractionFeatures, max_interactions=20))
    
    # # SequentialFeatures adapter (disabled for performance)
    # adapters.append(GlobalFeatureAdapter(SequentialFeatures))
    
    # # DistributionFeatures adapter (disabled for performance)
    # adapters.append(GlobalFeatureAdapter(DistributionFeatures))
    
    # # ClusteringFeatures adapter (disabled for performance)
    # adapters.append(GlobalFeatureAdapter(ClusteringFeatures))
    
    return adapters