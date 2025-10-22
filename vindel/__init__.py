"""
VINDEL: VINe-based DEgree-of-freedom Learning
An LLM-Integrated Framework for Synthetic IPD Generation

Enhanced with BMA, Constraint Enrichment, and Physics-Informed Learning
"""

from .core.constraints import (
    ConstraintCollection,
    MarginalConstraint,
    JointConstraint,
    ConditionalCorrelationConstraint,
    SurvivalConstraint,
    SubgroupSurvivalConstraint,
    TimeVaryingHRConstraint,
    CausalConstraint,
    MultiOutcomeConstraint,
    NetworkConstraint,
    PhysicsConstraint,
    SubgroupDefinition,
    DiseaseNaturalHistory,
    MonotonicityConstraint,
    PlausibilityBound,
    CorrelationSignConstraint,
)

from .bma.engine import (
    BMAEngine,
    BMAConfig,
    BMAResults,
    VineModelSpec,
    VineStructure,
    CopulaFamily,
)

from .integration.llm_retriever import (
    DiseaseProfileRetriever,
    DiseaseProfile,
    PaperConstraintExtractor,
    SmartParameterSelector,
    LLMToolInterface,
)

from .core.losses import (
    CompositeLoss,
    MarginalLoss,
    JointLoss,
    ConditionalCorrelationLoss,
    KaplanMeierLoss,
    SubgroupSurvivalLoss,
    CausalLoss,
    MultiOutcomeLoss,
)

__author__ = "Ahmed Azzam"

__all__ = [
    # Constraints
    "ConstraintCollection",
    "MarginalConstraint",
    "JointConstraint",
    "ConditionalCorrelationConstraint",
    "SurvivalConstraint",
    "SubgroupSurvivalConstraint",
    "CausalConstraint",
    "MultiOutcomeConstraint",
    "PhysicsConstraint",
    "SubgroupDefinition",
    
    # BMA
    "BMAEngine",
    "BMAConfig",
    "BMAResults",
    "VineModelSpec",
    "VineStructure",
    "CopulaFamily",
    
    # LLM Integration
    "DiseaseProfileRetriever",
    "DiseaseProfile",
    "PaperConstraintExtractor",
    "SmartParameterSelector",
    "LLMToolInterface",
    
    # Losses
    "CompositeLoss",
    "MarginalLoss",
    "JointLoss",
    "KaplanMeierLoss",
]
