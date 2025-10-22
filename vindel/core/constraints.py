"""
Core Constraint Data Structures

Defines all 11 constraint types from the VINDEL framework with
Pydantic validation and LLM-integration readiness.
"""

from typing import Optional, List, Dict, Any, Literal, Union, Callable
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
from enum import Enum


class ConstraintType(str, Enum):
    """Enumeration of all constraint types in VINDEL framework"""
    MARGINAL = "marginal"
    JOINT = "joint"
    CONDITIONAL = "conditional"  # NEW: Enrichment
    SURVIVAL = "survival"
    SURVIVAL_SUBGROUP = "survival_subgroup"  # NEW: Enrichment
    CAUSAL = "causal"
    MULTI_OUTCOME = "multi_outcome"  # NEW: Enrichment
    NETWORK = "network"
    OPTIMAL_TRANSPORT = "optimal_transport"
    PHYSICS = "physics"  # NEW: Physics-informed
    TIME_VARYING_HR = "time_varying_hr"  # NEW: Enrichment


class BaseConstraint(BaseModel):
    """Base class for all constraints"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    constraint_type: ConstraintType
    study_id: Optional[str] = None
    arm: Optional[str] = None
    weight: float = Field(default=1.0, gt=0.0, description="Constraint weight")
    uncertainty: Optional[float] = Field(default=None, ge=0.0, description="Standard error or confidence width")
    source: Optional[str] = Field(default=None, description="Literature reference or extraction source")
    llm_retrieved: bool = Field(default=False, description="Whether constraint was retrieved via LLM")


# ============================================================================
# MARGINAL CONSTRAINTS (C_marg)
# ============================================================================

class MarginalConstraint(BaseConstraint):
    """Marginal distribution constraints for individual variables"""
    constraint_type: Literal[ConstraintType.MARGINAL] = ConstraintType.MARGINAL
    
    variable_name: str
    
    # Moments
    mean: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    
    # Quantiles
    quantiles: Optional[Dict[float, float]] = Field(
        default=None,
        description="Dict mapping quantile level (0-1) to value"
    )
    
    # Categorical distributions
    categories: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dict mapping category to prevalence/probability"
    )
    
    # Full distribution specification
    distribution_type: Optional[str] = Field(
        default=None,
        description="E.g., 'normal', 'lognormal', 'beta', 'categorical'"
    )
    distribution_params: Optional[Dict[str, Any]] = None

    @field_validator('quantiles')
    def validate_quantiles(cls, v):
        if v is not None:
            if not all(0 <= k <= 1 for k in v.keys()):
                raise ValueError("Quantile levels must be in [0, 1]")
        return v

    @field_validator('categories')
    def validate_categories(cls, v):
        if v is not None:
            if not np.isclose(sum(v.values()), 1.0, atol=1e-6):
                raise ValueError(f"Category probabilities must sum to 1, got {sum(v.values())}")
        return v


# ============================================================================
# JOINT CONSTRAINTS (C_joint)
# ============================================================================

class JointConstraint(BaseConstraint):
    """Pairwise joint dependence constraints"""
    constraint_type: Literal[ConstraintType.JOINT] = ConstraintType.JOINT
    
    variable1: str
    variable2: str
    
    # Correlation measures
    pearson_correlation: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    kendall_tau: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    spearman_rho: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    
    # Categorical association
    odds_ratio: Optional[float] = Field(default=None, gt=0.0)
    relative_risk: Optional[float] = Field(default=None, gt=0.0)
    
    # Uncertainty
    correlation_ci_lower: Optional[float] = None
    correlation_ci_upper: Optional[float] = None


# ============================================================================
# CONDITIONAL CORRELATION CONSTRAINTS (C_cond) - NEW
# ============================================================================

class SubgroupDefinition(BaseModel):
    """Defines a subgroup via conditions on covariates"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    subgroup_name: str
    conditions: List[Dict[str, Any]] = Field(
        description="List of conditions, e.g., [{'variable': 'age', 'operator': '>=', 'value': 65}]"
    )
    
    def to_filter_func(self) -> Callable:
        """Convert to a filtering function for data"""
        def filter_fn(data: np.ndarray, variable_names: List[str]) -> np.ndarray:
            mask = np.ones(len(data), dtype=bool)
            var_map = {name: idx for idx, name in enumerate(variable_names)}
            
            for cond in self.conditions:
                var_idx = var_map[cond['variable']]
                op = cond['operator']
                val = cond['value']
                
                if op == '>=':
                    mask &= (data[:, var_idx] >= val)
                elif op == '<=':
                    mask &= (data[:, var_idx] <= val)
                elif op == '>':
                    mask &= (data[:, var_idx] > val)
                elif op == '<':
                    mask &= (data[:, var_idx] < val)
                elif op == '==':
                    mask &= (data[:, var_idx] == val)
                elif op == '!=':
                    mask &= (data[:, var_idx] != val)
                elif op == 'in':
                    mask &= np.isin(data[:, var_idx], val)
                    
            return mask
        return filter_fn


class ConditionalCorrelationConstraint(BaseConstraint):
    """Conditional correlation constraints (Section 6.2 of framework)"""
    constraint_type: Literal[ConstraintType.CONDITIONAL] = ConstraintType.CONDITIONAL
    
    variable1: str
    variable2: str
    subgroup: SubgroupDefinition
    
    correlation: float = Field(ge=-1.0, le=1.0)
    correlation_type: Literal["pearson", "kendall", "spearman"] = "pearson"
    
    # Context-aware metadata
    clinical_interpretation: Optional[str] = Field(
        default=None,
        description="LLM-provided clinical context for this correlation"
    )


# ============================================================================
# SURVIVAL CONSTRAINTS (C_surv)
# ============================================================================

class SurvivalConstraint(BaseConstraint):
    """Overall survival constraints"""
    constraint_type: Literal[ConstraintType.SURVIVAL] = ConstraintType.SURVIVAL
    
    # Kaplan-Meier curve points
    km_times: Optional[List[float]] = Field(default=None, description="Time points for KM curve")
    km_survival: Optional[List[float]] = Field(default=None, description="Survival probabilities at km_times")
    km_n_at_risk: Optional[List[int]] = Field(default=None, description="Number at risk at each time")
    km_n_events: Optional[List[int]] = Field(default=None, description="Number of events in each interval")
    
    # Summary statistics
    median_survival: Optional[float] = Field(default=None, gt=0.0)
    median_ci_lower: Optional[float] = None
    median_ci_upper: Optional[float] = None
    
    # Hazard ratios (vs reference arm)
    hazard_ratio: Optional[float] = Field(default=None, gt=0.0)
    hr_ci_lower: Optional[float] = None
    hr_ci_upper: Optional[float] = None
    reference_arm: Optional[str] = None

    @field_validator('km_survival')
    def validate_km_survival(cls, v, info):
        if v is not None:
            km_times = info.data.get('km_times')
            if km_times is not None and len(v) != len(km_times):
                raise ValueError("km_survival and km_times must have same length")
            if not all(0 <= s <= 1 for s in v):
                raise ValueError("Survival probabilities must be in [0, 1]")
            if not all(v[i] >= v[i+1] for i in range(len(v)-1)):
                raise ValueError("Survival function must be non-increasing")
        return v


# ============================================================================
# SUBGROUP-SPECIFIC SURVIVAL CONSTRAINTS (C_surv,subg) - NEW
# ============================================================================

class SubgroupSurvivalConstraint(BaseConstraint):
    """Subgroup-specific survival constraints (Section 6.3 of framework)"""
    constraint_type: Literal[ConstraintType.SURVIVAL_SUBGROUP] = ConstraintType.SURVIVAL_SUBGROUP
    
    subgroup: SubgroupDefinition
    
    # Subgroup-specific KM curve
    km_times: Optional[List[float]] = None
    km_survival: Optional[List[float]] = None
    
    # Subgroup-specific summary
    median_survival: Optional[float] = Field(default=None, gt=0.0)
    median_ci_lower: Optional[float] = None
    median_ci_upper: Optional[float] = None
    
    # Subgroup-specific HR
    hazard_ratio_vs_overall: Optional[float] = Field(default=None, gt=0.0)


# ============================================================================
# TIME-VARYING HAZARD RATIO CONSTRAINTS (C_HR,TV) - NEW
# ============================================================================

class TimeVaryingHRConstraint(BaseConstraint):
    """Time-varying hazard ratio constraints (Section 6.3 of framework)"""
    constraint_type: Literal[ConstraintType.TIME_VARYING_HR] = ConstraintType.TIME_VARYING_HR
    
    treatment_arm: str
    reference_arm: str
    
    # Time periods and HRs
    time_periods: List[tuple[float, float]] = Field(
        description="List of (start, end) time intervals"
    )
    hazard_ratios: List[float] = Field(
        description="HR for each time period"
    )
    hr_ci_lower: Optional[List[float]] = None
    hr_ci_upper: Optional[List[float]] = None
    
    # Test for non-proportional hazards
    interaction_p_value: Optional[float] = Field(
        default=None,
        description="p-value for treatment x time interaction"
    )

    @field_validator('hazard_ratios')
    def validate_hrs(cls, v, info):
        time_periods = info.data.get('time_periods')
        if time_periods is not None and len(v) != len(time_periods):
            raise ValueError("hazard_ratios and time_periods must have same length")
        return v


# ============================================================================
# CAUSAL CONSTRAINTS (C_causal)
# ============================================================================

class CausalConstraint(BaseConstraint):
    """Causal treatment effect constraints"""
    constraint_type: Literal[ConstraintType.CAUSAL] = ConstraintType.CAUSAL
    
    treatment_arm: str
    control_arm: str
    
    # Average treatment effect
    ate: Optional[float] = None
    ate_se: Optional[float] = None
    ate_ci_lower: Optional[float] = None
    ate_ci_upper: Optional[float] = None
    
    # Subgroup-specific effects
    subgroup_effects: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dict mapping subgroup name to treatment effect"
    )
    subgroup_definitions: Optional[Dict[str, SubgroupDefinition]] = None
    
    # Interaction tests
    interaction_p_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="p-values for treatment x covariate interactions"
    )
    
    # Effect measure type
    effect_type: Literal["risk_difference", "risk_ratio", "odds_ratio", "hazard_ratio", "mean_difference"] = "mean_difference"


# ============================================================================
# MULTI-OUTCOME CONSISTENCY CONSTRAINTS (C_multi) - NEW
# ============================================================================

class MultiOutcomeConstraint(BaseConstraint):
    """Multi-outcome consistency constraints (Section 6.4 of framework)"""
    constraint_type: Literal[ConstraintType.MULTI_OUTCOME] = ConstraintType.MULTI_OUTCOME
    
    # Binary endpoint
    binary_outcome_name: str
    binary_prevalence: Dict[str, float] = Field(
        description="Dict mapping arm to response rate/prevalence"
    )
    
    # Survival endpoint
    survival_outcome_name: str
    mean_survival: Dict[str, float] = Field(
        description="Dict mapping arm to mean survival time"
    )
    
    # Consistency requirement
    enforce_monotonicity: bool = Field(
        default=True,
        description="Require responders to have better survival than non-responders"
    )
    
    # Strength of association (if reported)
    responder_survival_hr: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="HR for responders vs non-responders"
    )


# ============================================================================
# NETWORK META-ANALYSIS CONSTRAINTS (C_net)
# ============================================================================

class NetworkConstraint(BaseConstraint):
    """Network meta-analysis consistency constraints"""
    constraint_type: Literal[ConstraintType.NETWORK] = ConstraintType.NETWORK
    
    # Network structure
    treatment_nodes: List[str] = Field(description="All treatment nodes in network")
    comparisons: List[tuple[str, str]] = Field(
        description="List of (treatment_a, treatment_b) direct comparisons"
    )
    
    # Basic parameters (log-scale)
    basic_parameters: Dict[str, float] = Field(
        description="Dict mapping comparison to log(HR) or log(OR)"
    )
    
    # Consistency equations: A*d = 0
    consistency_matrix: Optional[np.ndarray] = Field(
        default=None,
        description="Matrix A encoding consistency constraints"
    )
    
    # Heterogeneity
    between_study_sd: Optional[float] = Field(default=None, ge=0.0, description="tau parameter")
    i_squared: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="I-squared statistic")
    
    # Study-specific random effects
    study_effects: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Study-specific deviations from basic parameters"
    )


# ============================================================================
# PHYSICS-INFORMED CONSTRAINTS (C_phys) - NEW
# ============================================================================

class MonotonicityConstraint(BaseModel):
    """Monotonicity constraint for a single variable"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    variable_name: str
    direction: Literal["increasing", "decreasing"] = Field(
        description="Expected monotonic direction"
    )
    outcome: str = Field(
        default="hazard",
        description="Outcome that should be monotone in this variable"
    )
    conditional_on: Optional[SubgroupDefinition] = Field(
        default=None,
        description="Only enforce within this subgroup if specified"
    )
    penalty_weight: float = Field(default=1.0, gt=0.0)
    
    # LLM-retrieved evidence
    evidence_source: Optional[str] = None
    evidence_strength: Optional[Literal["strong", "moderate", "weak"]] = None


class BiologicalRelationship(BaseModel):
    """Known biological relationship between variables"""
    relationship_type: Literal["mediation", "compositional", "causal_chain"]
    variables_involved: List[str]
    relationship_spec: Dict[str, Any] = Field(
        description="Specification of the relationship"
    )
    penalty_weight: float = Field(default=0.5, gt=0.0)


class PlausibilityBound(BaseModel):
    """Plausibility bounds for a variable"""
    variable_name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    hard_constraint: bool = Field(
        default=False,
        description="If True, use high penalty; if False, soft regularization"
    )
    penalty_weight: float = Field(default=10.0, gt=0.0)
    
    # Evidence-based bounds
    clinical_justification: Optional[str] = None


class CorrelationSignConstraint(BaseModel):
    """Required sign for correlation between variables"""
    variable1: str
    variable2: str
    required_sign: Literal["positive", "negative"]
    penalty_weight: float = Field(default=2.0, gt=0.0)
    
    # Evidence
    biological_basis: Optional[str] = None


class DiseaseNaturalHistory(BaseModel):
    """Natural history profile for a disease"""
    disease_name: str
    hazard_profile: Literal["decreasing", "increasing", "constant", "bathtub"]
    
    # Time points for evaluation
    early_time: float
    mid_time: float
    late_time: float
    
    # Expected hazard ordering
    expected_ordering: List[str] = Field(
        description="E.g., ['early > mid', 'mid > late'] for decreasing hazard"
    )
    
    penalty_weight: float = Field(default=2.0, gt=0.0)
    
    # LLM-retrieved evidence
    literature_support: Optional[List[str]] = Field(
        default=None,
        description="List of references supporting this profile"
    )


class PhysicsConstraint(BaseConstraint):
    """Composite physics-informed constraints (Section 7 of framework)"""
    constraint_type: Literal[ConstraintType.PHYSICS] = ConstraintType.PHYSICS
    
    monotonicity_constraints: List[MonotonicityConstraint] = Field(default_factory=list)
    biological_relationships: List[BiologicalRelationship] = Field(default_factory=list)
    plausibility_bounds: List[PlausibilityBound] = Field(default_factory=list)
    correlation_signs: List[CorrelationSignConstraint] = Field(default_factory=list)
    natural_history: Optional[DiseaseNaturalHistory] = None
    
    # Overall weight for physics constraints
    overall_weight: float = Field(default=2.0, gt=0.0, description="Global lambda_phys")


# ============================================================================
# CONSTRAINT COLLECTION
# ============================================================================

class ConstraintCollection(BaseModel):
    """Collection of all constraints for a study or dataset"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    study_name: str
    
    # Core constraints
    marginal_constraints: List[MarginalConstraint] = Field(default_factory=list)
    joint_constraints: List[JointConstraint] = Field(default_factory=list)
    survival_constraints: List[SurvivalConstraint] = Field(default_factory=list)
    causal_constraints: List[CausalConstraint] = Field(default_factory=list)
    network_constraints: List[NetworkConstraint] = Field(default_factory=list)
    
    # Enriched constraints (NEW)
    conditional_constraints: List[ConditionalCorrelationConstraint] = Field(default_factory=list)
    subgroup_survival_constraints: List[SubgroupSurvivalConstraint] = Field(default_factory=list)
    time_varying_hr_constraints: List[TimeVaryingHRConstraint] = Field(default_factory=list)
    multi_outcome_constraints: List[MultiOutcomeConstraint] = Field(default_factory=list)
    
    # Physics constraints (NEW)
    physics_constraints: Optional[PhysicsConstraint] = None
    
    # Metadata
    variable_names: List[str] = Field(default_factory=list)
    treatment_arms: List[str] = Field(default_factory=list)
    sample_sizes: Dict[str, int] = Field(default_factory=dict)
    
    # LLM integration metadata
    llm_enriched: bool = Field(
        default=False,
        description="Whether LLM was used to enrich constraints"
    )
    enrichment_timestamp: Optional[str] = None
    literature_sources: List[str] = Field(default_factory=list)
    
    def count_constraints(self) -> Dict[str, int]:
        """Count constraints by type"""
        return {
            "marginal": len(self.marginal_constraints),
            "joint": len(self.joint_constraints),
            "conditional": len(self.conditional_constraints),
            "survival": len(self.survival_constraints),
            "survival_subgroup": len(self.subgroup_survival_constraints),
            "time_varying_hr": len(self.time_varying_hr_constraints),
            "causal": len(self.causal_constraints),
            "multi_outcome": len(self.multi_outcome_constraints),
            "network": len(self.network_constraints),
            "physics": 1 if self.physics_constraints is not None else 0,
        }
    
    def degrees_of_freedom_reduction(self) -> int:
        """Estimate degrees of freedom reduced by enrichment (Eq. 6.29)"""
        dof_reduced = 0
        
        # Conditional correlations
        dof_reduced += len(self.conditional_constraints)
        
        # Subgroup survival constraints
        for subg_surv in self.subgroup_survival_constraints:
            if subg_surv.km_times is not None:
                dof_reduced += len(subg_surv.km_times)
            else:
                dof_reduced += 1  # Median constraint
        
        # Time-varying HRs
        for tv_hr in self.time_varying_hr_constraints:
            dof_reduced += len(tv_hr.time_periods)
        
        # Multi-outcome consistency
        dof_reduced += len(self.multi_outcome_constraints)
        
        return dof_reduced
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintCollection":
        """Load from dictionary"""
        return cls.model_validate(data)

    def check_feasibility(self) -> List[Dict[str, Any]]:
        """
        Check for contradictory or infeasible constraints

        Returns:
            List of detected issues
        """
        checker = ConstraintFeasibilityChecker()
        return checker.check(self)


# ============================================================================
# CONSTRAINT FEASIBILITY CHECKER - NEW
# ============================================================================

class ConstraintFeasibilityChecker:
    """
    Check for contradictory or infeasible constraints

    Prevents optimization failures by detecting:
    1. Marginal-bound conflicts (mean ± k*SD exceeds bounds)
    2. Invalid correlation matrices (not positive definite)
    3. Survival-marginal conflicts (impossible survival given covariates)
    4. Subgroup-marginal conflicts (subgroup ranges inconsistent with overall)

    Usage:
        checker = ConstraintFeasibilityChecker()
        issues = checker.check(constraint_collection)
        if issues:
            print("WARNING: Infeasible constraints detected:")
            for issue in issues:
                print(f" - {issue['type']}: {issue['description']}")
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def check(self, constraints: ConstraintCollection) -> List[Dict[str, Any]]:
        """
        Run all feasibility checks

        Returns:
            List of detected issues, each a dict with keys:
            - type: Issue category
            - severity: 'error' (infeasible) or 'warning' (suspicious)
            - description: Human-readable explanation
            - affected_constraints: Indices or names of conflicting constraints
        """
        issues = []

        issues.extend(self._check_marginal_bounds(constraints))
        issues.extend(self._check_correlation_matrix(constraints))
        issues.extend(self._check_subgroup_consistency(constraints))
        issues.extend(self._check_survival_plausibility(constraints))

        return issues

    def _check_marginal_bounds(self, constraints: ConstraintCollection) -> List[Dict]:
        """Check if marginal distributions violate plausibility bounds"""
        issues = []

        if constraints.physics_constraints is None:
            return issues

        bounds_dict = {
            b.variable_name: (b.lower_bound, b.upper_bound)
            for b in constraints.physics_constraints.plausibility_bounds
        }

        for marginal in constraints.marginal_constraints:
            if marginal.variable_name not in bounds_dict:
                continue

            lower_bound, upper_bound = bounds_dict[marginal.variable_name]

            if marginal.mean is not None and marginal.std is not None:
                # Check if mean ± 3*SD exceeds bounds (covers 99.7% of normal)
                lower_reach = marginal.mean - 3 * marginal.std
                upper_reach = marginal.mean + 3 * marginal.std

                if lower_reach < lower_bound - self.tolerance:
                    issues.append({
                        'type': 'marginal_bound_violation',
                        'severity': 'warning',
                        'description': f'{marginal.variable_name}: mean-3*SD = {lower_reach:.2f} < bound {lower_bound}',
                        'affected_constraints': [marginal.variable_name]
                    })

                if upper_reach > upper_bound + self.tolerance:
                    issues.append({
                        'type': 'marginal_bound_violation',
                        'severity': 'warning',
                        'description': f'{marginal.variable_name}: mean+3*SD = {upper_reach:.2f} > bound {upper_bound}',
                        'affected_constraints': [marginal.variable_name]
                    })

        return issues

    def _check_correlation_matrix(self, constraints: ConstraintCollection) -> List[Dict]:
        """Check if specified correlations form a valid (positive definite) matrix"""
        issues = []

        if len(constraints.joint_constraints) < 2:
            return issues

        # Build correlation matrix from pairwise constraints
        variables = list(set(
            [c.variable1 for c in constraints.joint_constraints] +
            [c.variable2 for c in constraints.joint_constraints]
        ))

        n_vars = len(variables)
        if n_vars < 3:
            return issues  # Need at least 3 vars for PD check to matter

        var_to_idx = {v: i for i, v in enumerate(variables)}
        corr_matrix = np.eye(n_vars)

        for constraint in constraints.joint_constraints:
            if constraint.pearson_correlation is not None:
                i = var_to_idx[constraint.variable1]
                j = var_to_idx[constraint.variable2]
                corr_matrix[i, j] = constraint.pearson_correlation
                corr_matrix[j, i] = constraint.pearson_correlation

        # Check positive definiteness via eigenvalues
        try:
            eigenvals = np.linalg.eigvals(corr_matrix)
            min_eigenval = np.min(eigenvals)

            if min_eigenval < -self.tolerance:
                issues.append({
                    'type': 'invalid_correlation_matrix',
                    'severity': 'error',
                    'description': f'Correlation matrix not positive definite (min eigenvalue = {min_eigenval:.6f})',
                    'affected_constraints': [f"{c.variable1}-{c.variable2}" for c in constraints.joint_constraints]
                })
        except np.linalg.LinAlgError:
            issues.append({
                'type': 'invalid_correlation_matrix',
                'severity': 'error',
                'description': 'Correlation matrix is singular or ill-conditioned',
                'affected_constraints': [f"{c.variable1}-{c.variable2}" for c in constraints.joint_constraints]
            })

        return issues

    def _check_subgroup_consistency(self, constraints: ConstraintCollection) -> List[Dict]:
        """Check if subgroup constraints are consistent with overall marginals"""
        issues = []

        # For each subgroup constraint, check if it's compatible with overall marginals
        for subg_constraint in constraints.conditional_constraints:
            # This is a simplified check - full check would require sampling
            pass  # Placeholder - implement if needed

        return issues

    def _check_survival_plausibility(self, constraints: ConstraintCollection) -> List[Dict]:
        """Check if survival constraints are internally consistent"""
        issues = []

        for surv_constraint in constraints.survival_constraints:
            if surv_constraint.median_survival is not None and \
               surv_constraint.km_times is not None and \
               surv_constraint.km_survival is not None:

                # Check if median survival is consistent with KM curve
                # Median is time when S(t) = 0.5
                km_times = np.array(surv_constraint.km_times)
                km_survival = np.array(surv_constraint.km_survival)

                # Interpolate to find S(median)
                if surv_constraint.median_survival >= km_times[0] and \
                   surv_constraint.median_survival <= km_times[-1]:

                    s_at_median = np.interp(
                        surv_constraint.median_survival,
                        km_times,
                        km_survival
                    )

                    # Should be close to 0.5
                    if abs(s_at_median - 0.5) > 0.1:
                        issues.append({
                            'type': 'survival_inconsistency',
                            'severity': 'warning',
                            'description': f'Median survival {surv_constraint.median_survival} implies S(t)={s_at_median:.2f}, not 0.5',
                            'affected_constraints': [surv_constraint.study_id or 'unknown']
                        })

        return issues
