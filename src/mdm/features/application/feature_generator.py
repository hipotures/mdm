"""Application service for orchestrating feature generation."""

import time
from typing import Dict, Optional, Any
import pandas as pd

from loguru import logger

from mdm.features.domain import (
    FeatureService,
    IFeatureRepository,
    IFeatureLoader,
    ICustomFeatureLoader,
    IFeatureWriter,
    FeatureGenerationConfig,
    FeatureSet
)

from .dtos import (
    FeatureGenerationRequest,
    FeatureGenerationResponse,
    FeatureGenerationMode,
    BatchProcessingRequest,
    BatchProcessingResponse
)


class FeatureGenerationUseCase:
    """Application service that orchestrates feature generation.
    
    This class coordinates between domain services and infrastructure,
    handling the overall flow of feature generation.
    """
    
    def __init__(self,
                 feature_service: FeatureService,
                 feature_loader: IFeatureLoader,
                 feature_writer: IFeatureWriter,
                 custom_loader: ICustomFeatureLoader,
                 feature_repository: IFeatureRepository):
        """Initialize the use case with required services.
        
        Args:
            feature_service: Domain service for feature generation
            feature_loader: Infrastructure service for loading data
            feature_writer: Infrastructure service for writing features
            custom_loader: Infrastructure service for loading custom features
            feature_repository: Repository for feature metadata
        """
        self.feature_service = feature_service
        self.feature_loader = feature_loader
        self.feature_writer = feature_writer
        self.custom_loader = custom_loader
        self.feature_repository = feature_repository
    
    def execute(self, request: FeatureGenerationRequest) -> FeatureGenerationResponse:
        """Execute feature generation based on request.
        
        Args:
            request: Feature generation request
            
        Returns:
            Feature generation response
        """
        start_time = time.time()
        feature_tables = {}
        total_features = 0
        features_per_table = {}
        
        try:
            # Load custom features if enabled
            custom_features = None
            if request.enable_custom_features:
                custom_features = self.custom_loader.load(request.dataset_name)
                if custom_features:
                    logger.info(f"Loaded custom features for {request.dataset_name}")
            
            # Process each table
            for table_type, table_name in request.table_names.items():
                logger.info(f"Generating features for {table_type} table: {table_name}")
                
                if request.mode == FeatureGenerationMode.FULL:
                    feature_count = self._process_table_full(
                        request, table_name, table_type, custom_features
                    )
                elif request.mode == FeatureGenerationMode.INCREMENTAL:
                    feature_count = self._process_table_incremental(
                        request, table_name, table_type, custom_features
                    )
                else:  # SELECTED_COLUMNS
                    feature_count = self._process_selected_columns(
                        request, table_name, table_type, custom_features
                    )
                
                feature_table_name = f"{table_type}_features"
                feature_tables[table_type] = feature_table_name
                features_per_table[table_type] = feature_count
                total_features += feature_count
            
            generation_time = time.time() - start_time
            
            return FeatureGenerationResponse(
                dataset_name=request.dataset_name,
                feature_tables=feature_tables,
                total_features_generated=total_features,
                features_per_table=features_per_table,
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}", exc_info=True)
            return FeatureGenerationResponse(
                dataset_name=request.dataset_name,
                feature_tables={},
                total_features_generated=0,
                features_per_table={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _process_table_full(self,
                           request: FeatureGenerationRequest,
                           table_name: str,
                           table_type: str,
                           custom_features: Optional[Any]) -> int:
        """Process a table in full mode with batching.
        
        Args:
            request: Feature generation request
            table_name: Source table name
            table_type: Type of table (train, test, etc.)
            custom_features: Optional custom feature generator
            
        Returns:
            Number of features generated
        """
        feature_table_name = f"{table_type}_features"
        total_features = 0
        batch_number = 0
        offset = 0
        
        # Create feature generation config
        config = FeatureGenerationConfig(
            dataset_name=request.dataset_name,
            target_column=request.target_column,
            id_columns=request.id_columns,
            column_types=request.column_types
        )
        
        while True:
            # Create batch request
            batch_request = BatchProcessingRequest(
                dataset_name=request.dataset_name,
                source_table=table_name,
                target_table=feature_table_name,
                column_types=request.column_types,
                batch_size=request.batch_size,
                offset=offset
            )
            
            # Process batch
            batch_response = self._process_batch(
                batch_request, config, custom_features, 
                is_first_batch=(batch_number == 0)
            )
            
            total_features = max(total_features, batch_response.features_generated)
            batch_number += 1
            
            if not batch_response.has_more:
                break
            
            offset = batch_response.next_offset
        
        # Save metadata
        metadata = self.feature_repository.load_metadata(request.dataset_name)
        if metadata:
            logger.info(f"Saved metadata for {len(metadata)} features")
        
        return total_features
    
    def _process_batch(self,
                      request: BatchProcessingRequest,
                      config: FeatureGenerationConfig,
                      custom_features: Optional[Any],
                      is_first_batch: bool = False) -> BatchProcessingResponse:
        """Process a single batch of data.
        
        Args:
            request: Batch processing request
            config: Feature generation configuration
            custom_features: Optional custom feature generator
            is_first_batch: Whether this is the first batch
            
        Returns:
            Batch processing response
        """
        start_time = time.time()
        
        # Load batch data
        df = self.feature_loader.load_data(request.dataset_name, request.source_table)
        
        if df is None or df.empty:
            return BatchProcessingResponse(
                batch_number=request.offset // request.batch_size,
                rows_processed=0,
                features_generated=0,
                processing_time=time.time() - start_time,
                has_more=False
            )
        
        # Generate features
        feature_set = self.feature_service.generate_features(
            df, config, custom_features
        )
        
        # Write features
        self.feature_writer.write_features_batch(
            request.dataset_name,
            request.target_table,
            feature_set.features,
            request.offset // request.batch_size,
            is_first_batch=is_first_batch
        )
        
        # Save metadata (only for first batch)
        if is_first_batch:
            self.feature_repository.save_metadata(
                request.dataset_name,
                feature_set.metadata
            )
        
        processing_time = time.time() - start_time
        
        return BatchProcessingResponse(
            batch_number=request.offset // request.batch_size,
            rows_processed=len(df),
            features_generated=feature_set.feature_count,
            processing_time=processing_time,
            has_more=len(df) == request.batch_size,
            next_offset=request.offset + len(df)
        )
    
    def _process_table_incremental(self,
                                  request: FeatureGenerationRequest,
                                  table_name: str,
                                  table_type: str,
                                  custom_features: Optional[Any]) -> int:
        """Process a table in incremental mode.
        
        This mode only processes new or updated data.
        
        Args:
            request: Feature generation request
            table_name: Source table name
            table_type: Type of table
            custom_features: Optional custom feature generator
            
        Returns:
            Number of features generated
        """
        # TODO: Implement incremental processing
        # For now, fall back to full processing
        logger.warning("Incremental mode not yet implemented, using full mode")
        return self._process_table_full(request, table_name, table_type, custom_features)
    
    def _process_selected_columns(self,
                                 request: FeatureGenerationRequest,
                                 table_name: str,
                                 table_type: str,
                                 custom_features: Optional[Any]) -> int:
        """Process only selected columns.
        
        Args:
            request: Feature generation request
            table_name: Source table name
            table_type: Type of table
            custom_features: Optional custom feature generator
            
        Returns:
            Number of features generated
        """
        # Load full data (could be optimized to load only selected columns)
        df = self.feature_loader.load_data(request.dataset_name, table_name)
        
        if df is None or df.empty:
            return 0
        
        # Filter column types for selected columns
        selected_column_types = {
            col: col_type 
            for col, col_type in request.column_types.items()
            if col in request.selected_columns
        }
        
        # Generate features only for selected columns
        feature_set = self.feature_service.generate_features_for_columns(
            df,
            request.selected_columns,
            selected_column_types
        )
        
        # Write features
        feature_table_name = f"{table_type}_features_selected"
        self.feature_writer.write_features(
            request.dataset_name,
            feature_table_name,
            feature_set.features,
            mode="replace"
        )
        
        # Save metadata
        self.feature_repository.save_metadata(
            request.dataset_name,
            feature_set.metadata
        )
        
        return feature_set.feature_count