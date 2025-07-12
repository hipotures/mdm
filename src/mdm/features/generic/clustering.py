"""Clustering-based features generator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from mdm.features.base_global import GlobalFeatureOperation
from mdm.features.utils import check_signal


class ClusteringFeatures(GlobalFeatureOperation):
    """Generate features using unsupervised clustering algorithms.
    
    Creates cluster assignments and anomaly scores using various
    clustering and outlier detection methods.
    """
    
    def __init__(
        self, 
        k_values: Optional[List[int]] = None,
        use_minibatch: bool = True,
        sample_size: int = 10000,
        random_state: int = 42
    ):
        """Initialize clustering feature generator.
        
        Args:
            k_values: List of k values for k-means (default: [5, 10])
            use_minibatch: Use MiniBatchKMeans for large datasets
            sample_size: Sample size for expensive operations
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.k_values = k_values or [5, 10]
        self.use_minibatch = use_minibatch
        self.sample_size = sample_size
        self.random_state = random_state
        
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns suitable for clustering.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of numeric column names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns with variation
        applicable = []
        for col in numeric_cols:
            if df[col].std() > 0 and df[col].nunique() > 5:
                applicable.append(col)
        
        logger.debug(f"Found {len(applicable)} columns suitable for clustering")
        return applicable
    
    def _prepare_data(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Prepare and scale data for clustering.
        
        Args:
            df: Input dataframe
            columns: Column names to use
            
        Returns:
            Scaled numpy array
        """
        # Handle missing values
        data = df[columns].fillna(df[columns].mean())
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data
    
    def _generate_kmeans_features(
        self, 
        data: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate k-means clustering features.
        
        Args:
            data: Scaled data array
            k: Number of clusters
            
        Returns:
            Tuple of (cluster assignments, distances to centroids)
        """
        n_samples = data.shape[0]
        
        # Use MiniBatchKMeans for large datasets
        if self.use_minibatch and n_samples > 50000:
            model = MiniBatchKMeans(
                n_clusters=k, 
                random_state=self.random_state,
                batch_size=1000
            )
        else:
            model = KMeans(
                n_clusters=k, 
                random_state=self.random_state,
                n_init=10
            )
        
        # Fit model
        cluster_labels = model.fit_predict(data)
        
        # Calculate distances to cluster centers
        distances = model.transform(data).min(axis=1)
        
        return cluster_labels, distances
    
    def _generate_isolation_features(self, data: np.ndarray) -> np.ndarray:
        """Generate Isolation Forest anomaly scores.
        
        Args:
            data: Scaled data array
            
        Returns:
            Array of anomaly scores
        """
        n_samples = data.shape[0]
        
        # Sample for large datasets
        if n_samples > self.sample_size:
            sample_idx = np.random.RandomState(self.random_state).choice(
                n_samples, size=self.sample_size, replace=False
            )
            sample_data = data[sample_idx]
        else:
            sample_data = data
        
        # Fit Isolation Forest
        model = IsolationForest(
            contamination=0.1,
            random_state=self.random_state,
            n_estimators=100
        )
        model.fit(sample_data)
        
        # Get anomaly scores for all data
        scores = model.score_samples(data)
        
        return scores
    
    def _generate_dbscan_features(self, data: np.ndarray) -> np.ndarray:
        """Generate DBSCAN clustering features.
        
        Args:
            data: Scaled data array
            
        Returns:
            Array of cluster labels (-1 for noise)
        """
        n_samples = data.shape[0]
        
        # Use sampling for large datasets
        if n_samples > self.sample_size:
            # For large datasets, use a density-based sampling approach
            logger.info(f"Dataset too large for DBSCAN ({n_samples} samples), using sampling")
            sample_idx = np.random.RandomState(self.random_state).choice(
                n_samples, size=self.sample_size, replace=False
            )
            
            # Fit on sample
            model = DBSCAN(eps=0.5, min_samples=5)
            sample_labels = model.fit_predict(data[sample_idx])
            
            # Approximate labels for full dataset
            labels = np.full(n_samples, -1)  # Default to noise
            labels[sample_idx] = sample_labels
            
            return labels
        
        # For smaller datasets, use full DBSCAN
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(data)
        
        return labels
    
    def _generate_lof_features(self, data: np.ndarray) -> np.ndarray:
        """Generate Local Outlier Factor scores.
        
        Args:
            data: Scaled data array
            
        Returns:
            Array of LOF scores
        """
        n_samples = data.shape[0]
        
        # Use sampling for large datasets
        if n_samples > self.sample_size:
            logger.info(f"Dataset too large for LOF ({n_samples} samples), using sampling")
            sample_idx = np.random.RandomState(self.random_state).choice(
                n_samples, size=self.sample_size, replace=False
            )
            
            # Fit on sample and predict
            model = LocalOutlierFactor(n_neighbors=20, novelty=True)
            model.fit(data[sample_idx])
            
            # Get scores for all data
            scores = model.score_samples(data)
        else:
            # For smaller datasets, use full LOF
            model = LocalOutlierFactor(n_neighbors=20)
            model.fit(data)
            scores = model.negative_outlier_factor_
        
        return scores
    
    def generate_features(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate all clustering features.
        
        Args:
            df: Input dataframe
            target_column: Target column to exclude
            id_columns: ID columns to exclude
            
        Returns:
            Tuple of (features dataframe, feature descriptions)
        """
        features = pd.DataFrame(index=df.index)
        descriptions = {}
        
        # Get applicable columns
        columns = self.get_applicable_columns(df)
        
        # Exclude target and ID columns
        columns = [col for col in columns 
                  if col != target_column and col not in (id_columns or [])]
        
        if len(columns) < 2:
            logger.warning("Not enough columns for clustering")
            return features, descriptions
        
        # Limit columns to prevent memory issues
        if len(columns) > 20:
            # Select top columns by variance
            variances = df[columns].var()
            columns = variances.nlargest(20).index.tolist()
            logger.info(f"Limited to top 20 columns by variance for clustering")
        
        # Prepare data
        data = self._prepare_data(df, columns)
        
        # K-means clustering
        for k in self.k_values:
            try:
                labels, distances = self._generate_kmeans_features(data, k)
                
                features[f'kmeans_cluster_{k}'] = labels
                descriptions[f'kmeans_cluster_{k}'] = f"K-means cluster assignment (k={k})"
                
                features[f'kmeans_distance_{k}'] = distances
                descriptions[f'kmeans_distance_{k}'] = f"Distance to nearest centroid (k={k})"
            except Exception as e:
                logger.warning(f"Failed to generate k-means features (k={k}): {e}")
        
        # Isolation Forest
        try:
            iso_scores = self._generate_isolation_features(data)
            features['isolation_score'] = iso_scores
            descriptions['isolation_score'] = "Isolation Forest anomaly score"
            
            features['is_anomaly'] = (iso_scores < np.percentile(iso_scores, 10)).astype(int)
            descriptions['is_anomaly'] = "Binary anomaly flag (bottom 10% isolation score)"
        except Exception as e:
            logger.warning(f"Failed to generate Isolation Forest features: {e}")
        
        # DBSCAN (only for smaller datasets)
        if len(df) <= 50000:
            try:
                dbscan_labels = self._generate_dbscan_features(data)
                features['density_cluster'] = dbscan_labels
                descriptions['density_cluster'] = "DBSCAN cluster assignment (-1 for noise)"
                
                features['is_noise_point'] = (dbscan_labels == -1).astype(int)
                descriptions['is_noise_point'] = "Binary flag for DBSCAN noise points"
            except Exception as e:
                logger.warning(f"Failed to generate DBSCAN features: {e}")
        
        # Local Outlier Factor (only for smaller datasets)
        if len(df) <= 20000:
            try:
                lof_scores = self._generate_lof_features(data)
                features['lof_score'] = lof_scores
                descriptions['lof_score'] = "Local Outlier Factor score"
            except Exception as e:
                logger.warning(f"Failed to generate LOF features: {e}")
        
        # Apply signal check
        features, descriptions = check_signal(features, descriptions, self.min_signal_ratio)
        
        logger.info(f"Generated {len(features.columns)} clustering features")
        return features, descriptions