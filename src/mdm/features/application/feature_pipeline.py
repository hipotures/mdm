"""Pipeline for orchestrating feature generation workflow."""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from loguru import logger

from mdm.features.domain import FeatureSet
from .dtos import FeatureGenerationRequest, FeatureGenerationResponse


@dataclass
class PipelineContext:
    """Context object passed through pipeline steps."""
    request: FeatureGenerationRequest
    response: Optional[FeatureGenerationResponse] = None
    intermediate_results: Dict[str, Any] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        """Initialize intermediate results."""
        if self.intermediate_results is None:
            self.intermediate_results = {}
    
    def add_result(self, key: str, value: Any) -> None:
        """Add an intermediate result."""
        self.intermediate_results[key] = value
    
    def get_result(self, key: str) -> Optional[Any]:
        """Get an intermediate result."""
        return self.intermediate_results.get(key)
    
    def has_error(self) -> bool:
        """Check if pipeline has encountered an error."""
        return self.error is not None


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    @abstractmethod
    def name(self) -> str:
        """Get step name for logging."""
        pass
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the pipeline step.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context
        """
        pass
    
    def can_execute(self, context: PipelineContext) -> bool:
        """Check if step can be executed.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if step can execute
        """
        return not context.has_error()


class ValidationStep(PipelineStep):
    """Step for validating the feature generation request."""
    
    def name(self) -> str:
        return "RequestValidation"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Validate the request."""
        try:
            request = context.request
            
            # Validate table names
            if not request.table_names:
                raise ValueError("No tables specified for feature generation")
            
            # Validate column types
            if not request.column_types:
                raise ValueError("No column types specified")
            
            # Validate mode-specific requirements
            if request.mode.value == "selected_columns" and not request.selected_columns:
                raise ValueError("No columns selected for selected_columns mode")
            
            logger.info(f"Request validation passed for dataset {request.dataset_name}")
            context.add_result("validation_passed", True)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            context.error = e
        
        return context


class DataLoadingStep(PipelineStep):
    """Step for loading data information."""
    
    def __init__(self, data_loader: Any):
        self.data_loader = data_loader
    
    def name(self) -> str:
        return "DataLoading"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Load data information."""
        try:
            # Here we would typically load metadata about the data
            # For now, we'll just pass through
            context.add_result("data_loaded", True)
            logger.info("Data loading step completed")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            context.error = e
        
        return context


class FeatureGenerationStep(PipelineStep):
    """Step for actual feature generation."""
    
    def __init__(self, feature_generator: Any):
        self.feature_generator = feature_generator
    
    def name(self) -> str:
        return "FeatureGeneration"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generate features."""
        try:
            # Execute feature generation
            response = self.feature_generator.execute(context.request)
            context.response = response
            
            if response.success:
                logger.info(f"Generated {response.total_features_generated} features")
            else:
                logger.error(f"Feature generation failed: {response.error_message}")
                
        except Exception as e:
            logger.error(f"Feature generation step failed: {e}")
            context.error = e
        
        return context


class MetadataPersistenceStep(PipelineStep):
    """Step for persisting feature metadata."""
    
    def __init__(self, metadata_repository: Any):
        self.metadata_repository = metadata_repository
    
    def name(self) -> str:
        return "MetadataPersistence"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Persist metadata."""
        try:
            if context.response and context.response.success:
                # Metadata is already saved in the feature generation step
                # This step could do additional metadata processing
                logger.info("Metadata persistence completed")
                
        except Exception as e:
            logger.error(f"Metadata persistence failed: {e}")
            # Non-critical error, don't fail the pipeline
            if context.response:
                context.response.add_warning(f"Metadata persistence failed: {e}")
        
        return context


class FeatureGenerationPipeline:
    """Pipeline for managing feature generation workflow."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.steps: List[PipelineStep] = []
        self.error_handlers: List[Callable[[PipelineContext], None]] = []
    
    def add_step(self, step: PipelineStep) -> 'FeatureGenerationPipeline':
        """Add a step to the pipeline.
        
        Args:
            step: Pipeline step to add
            
        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self
    
    def add_error_handler(self, handler: Callable[[PipelineContext], None]) -> 'FeatureGenerationPipeline':
        """Add an error handler.
        
        Args:
            handler: Error handler function
            
        Returns:
            Self for chaining
        """
        self.error_handlers.append(handler)
        return self
    
    def execute(self, request: FeatureGenerationRequest) -> FeatureGenerationResponse:
        """Execute the pipeline.
        
        Args:
            request: Feature generation request
            
        Returns:
            Feature generation response
        """
        start_time = time.time()
        context = PipelineContext(request=request)
        
        # Execute each step
        for step in self.steps:
            if step.can_execute(context):
                logger.info(f"Executing pipeline step: {step.name()}")
                context = step.execute(context)
            else:
                logger.warning(f"Skipping step {step.name()} due to pipeline error")
        
        # Handle any errors
        if context.has_error():
            for handler in self.error_handlers:
                handler(context)
        
        # Return response or create error response
        if context.response:
            return context.response
        else:
            return FeatureGenerationResponse(
                dataset_name=request.dataset_name,
                feature_tables={},
                total_features_generated=0,
                features_per_table={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(context.error) if context.error else "Unknown error"
            )


class FeaturePipelineBuilder:
    """Builder for creating feature generation pipelines."""
    
    def __init__(self):
        """Initialize builder."""
        self.pipeline = FeatureGenerationPipeline()
    
    def with_validation(self) -> 'FeaturePipelineBuilder':
        """Add validation step."""
        self.pipeline.add_step(ValidationStep())
        return self
    
    def with_data_loading(self, data_loader: Any) -> 'FeaturePipelineBuilder':
        """Add data loading step."""
        self.pipeline.add_step(DataLoadingStep(data_loader))
        return self
    
    def with_feature_generation(self, feature_generator: Any) -> 'FeaturePipelineBuilder':
        """Add feature generation step."""
        self.pipeline.add_step(FeatureGenerationStep(feature_generator))
        return self
    
    def with_metadata_persistence(self, metadata_repository: Any) -> 'FeaturePipelineBuilder':
        """Add metadata persistence step."""
        self.pipeline.add_step(MetadataPersistenceStep(metadata_repository))
        return self
    
    def with_error_handler(self, handler: Callable[[PipelineContext], None]) -> 'FeaturePipelineBuilder':
        """Add error handler."""
        self.pipeline.add_error_handler(handler)
        return self
    
    def build(self) -> FeatureGenerationPipeline:
        """Build the pipeline."""
        return self.pipeline