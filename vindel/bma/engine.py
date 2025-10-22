"""
Bayesian Model Averaging (BMA) Engine

Implements Algorithm 1 from Section 5 of VINDEL framework:
- Phase 1: Model selection via short burn-in and BIC computation
- Phase 2: Proportional posterior sampling with full training
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VineStructure(str, Enum):
    """Vine copula structure types"""
    C_VINE = "c_vine"
    D_VINE = "d_vine"
    R_VINE = "r_vine"
    INDEPENDENCE = "independence"


class CopulaFamily(str, Enum):
    """Copula family types"""
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"


@dataclass
class VineModelSpec:
    """Specification for a vine copula model"""
    model_id: str
    structure: VineStructure
    families: List[CopulaFamily]
    family_penalty: float = 0.0  # Lambda controlling complexity
    
    def __hash__(self):
        return hash((self.model_id, self.structure, tuple(self.families)))
    
    def count_parameters(self, n_variables: int) -> int:
        """Estimate number of parameters (k_m for BIC)"""
        n_pairs = n_variables * (n_variables - 1) // 2
        
        # Base copula parameters
        params_per_family = {
            CopulaFamily.GAUSSIAN: 1,  # correlation
            CopulaFamily.STUDENT_T: 2,  # correlation + df
            CopulaFamily.CLAYTON: 1,  # theta
            CopulaFamily.GUMBEL: 1,  # theta
            CopulaFamily.FRANK: 1,  # theta
        }
        
        avg_params = np.mean([params_per_family[f] for f in self.families])
        total_params = int(n_pairs * avg_params)
        
        # Add marginal transformation parameters (quantile warping)
        # Approximate: 10 knot points per variable
        total_params += n_variables * 10
        
        return total_params


@dataclass
class BMAConfig:
    """Configuration for BMA training"""
    # Model space
    model_specs: List[VineModelSpec]
    
    # Training hyperparameters
    n_posterior_draws: int = 50
    short_burnin_epochs: int = 50
    full_training_epochs: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 512
    
    # BMA parameters
    min_model_weight: float = 0.01  # Prune models below this
    use_uniform_prior: bool = True
    
    # Computational
    random_seed_base: int = 42
    n_parallel_jobs: int = 1
    device: str = "cpu"
    
    # Convergence
    convergence_tolerance: float = 1e-4
    convergence_window: int = 10
    early_stopping: bool = True
    
    # Validation
    validate_model_weights: bool = True
    min_effective_models: int = 3


@dataclass
class ModelPosterior:
    """Posterior results for a single model"""
    model_spec: VineModelSpec
    bic: float
    weight: float
    n_parameters: int
    final_loss: float
    converged: bool
    training_time_seconds: float
    
    # Store trained parameters (placeholder - actual implementation would store full model)
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class BMAResults:
    """Results from BMA training"""
    model_posteriors: List[ModelPosterior]
    total_draws: int
    draws_per_model: Dict[str, int]
    
    # Variance decomposition (Eq. 5.13-5.16)
    within_draw_variance: float = 0.0  # W_bar
    within_model_variance: float = 0.0  # B_within
    between_model_variance: float = 0.0  # B_between
    total_variance: float = 0.0  # T_BMA
    
    # Structural uncertainty diagnostics
    structural_uncertainty_proportion: float = 0.0  # pi_struct
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    training_time_total_seconds: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "n_models": len(self.model_posteriors),
            "effective_models": sum(1 for m in self.model_posteriors if m.weight >= 0.05),
            "top_model_weight": max(m.weight for m in self.model_posteriors),
            "model_weights": {m.model_spec.model_id: m.weight for m in self.model_posteriors},
            "structural_uncertainty": {
                "proportion": self.structural_uncertainty_proportion,
                "interpretation": self._interpret_structural_uncertainty()
            },
            "variance_decomposition": {
                "within_draw": self.within_draw_variance,
                "within_model": self.within_model_variance,
                "between_model": self.between_model_variance,
                "total": self.total_variance
            }
        }
    
    def _interpret_structural_uncertainty(self) -> str:
        """Interpret structural uncertainty level (Section 5.4.2)"""
        pi = self.structural_uncertainty_proportion
        if pi < 0.10:
            return "Negligible - single model acceptable"
        elif pi < 0.30:
            return "Moderate - BMA recommended"
        else:
            return "High - BMA essential, estimand structurally sensitive"

    def compute_effective_sample_size(self) -> float:
        """
        Compute effective sample size (ESS) of model weights

        ESS = 1 / sum(w_m^2)

        Interpretation:
        - ESS = 1: Single model dominates
        - ESS = M: Uniform weights (maximum uncertainty)
        - ESS < 3: Concerning - model space may be too narrow
        """
        weights = np.array([mp.weight for mp in self.model_posteriors])
        ess = 1.0 / np.sum(weights ** 2)
        return ess

    def get_model_diversity_metrics(self) -> Dict[str, float]:
        """
        Compute diversity metrics for model ensemble

        Returns:
            Dict with metrics:
            - effective_sample_size: ESS of weights
            - entropy: Shannon entropy of weight distribution
            - max_weight: Weight of top model
            - gini_coefficient: Inequality in weight distribution
        """
        weights = np.array([mp.weight for mp in self.model_posteriors])
        weights = weights / weights.sum()  # Normalize

        # Effective sample size
        ess = 1.0 / np.sum(weights ** 2)

        # Shannon entropy
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights))  # Uniform distribution
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Max weight
        max_weight = np.max(weights)

        # Gini coefficient
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n

        return {
            'effective_sample_size': ess,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'max_weight': max_weight,
            'gini_coefficient': gini,
            'interpretation': self._interpret_diversity(ess, max_weight)
        }

    def _interpret_diversity(self, ess: float, max_weight: float) -> str:
        """Interpret diversity metrics"""
        if max_weight > 0.8:
            return f"Single model dominates (ESS={ess:.1f}) - BMA provides honest uncertainty but limited diversity"
        elif ess < 3.0:
            return f"Low diversity (ESS={ess:.1f}) - consider expanding model space"
        elif ess > len(self.model_posteriors) * 0.7:
            return f"High diversity (ESS={ess:.1f}) - substantial structural uncertainty"
        else:
            return f"Moderate diversity (ESS={ess:.1f}) - reasonable model disagreement"


class BMAEngine:
    """
    Bayesian Model Averaging Engine for VINDEL
    
    Implements two-phase algorithm from Section 5.3:
    Phase 1: Model screening with short burn-in
    Phase 2: Proportional posterior sampling
    """
    
    def __init__(self, config: BMAConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: Optional[BMAResults] = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed_base)
        np.random.seed(config.random_seed_base)
    
    def get_default_model_space(
        self,
        n_variables: int,
        constraints: Optional[Any] = None
    ) -> List[VineModelSpec]:
        """
        Generate context-aware model space based on problem characteristics

        Enhanced version that considers:
        - Dimensionality (fewer models for high-D)
        - Constraint patterns (survival-heavy → tail copulas)
        - Correlation strength (high corr → Gaussian bias)

        Args:
            n_variables: Number of variables
            constraints: Optional ConstraintCollection for smart selection

        Returns:
            List of VineModelSpec objects
        """
        models = []

        # Analyze constraints if provided
        has_heavy_survival = False
        has_high_correlations = False
        has_tail_dependence = False

        if constraints is not None:
            # Check for survival constraints
            has_heavy_survival = len(constraints.survival_constraints) > 3

            # Check for high correlations
            if len(constraints.joint_constraints) > 0:
                max_corr = max(
                    abs(c.pearson_correlation)
                    for c in constraints.joint_constraints
                    if c.pearson_correlation is not None
                )
                has_high_correlations = max_corr > 0.6

            # Check for conditional correlations (suggests tail dependence)
            has_tail_dependence = len(constraints.conditional_constraints) > 0

        # Base models (always include)
        models.append(
            VineModelSpec("m1_cvine_gaussian", VineStructure.C_VINE, [CopulaFamily.GAUSSIAN])
        )

        # Add Student-t if high correlations (heavier tails)
        if has_high_correlations or n_variables <= 15:
            models.append(
                VineModelSpec("m2_cvine_student_t", VineStructure.C_VINE, [CopulaFamily.STUDENT_T])
            )

        # Add Archimedean copulas for survival-heavy problems
        if has_heavy_survival:
            models.extend([
                VineModelSpec("m3_cvine_clayton", VineStructure.C_VINE, [CopulaFamily.CLAYTON]),
                VineModelSpec("m4_cvine_gumbel", VineStructure.C_VINE, [CopulaFamily.GUMBEL]),
            ])

        # Add mixed family for robustness (if not too high-dimensional)
        if n_variables <= 20:
            mixed_families = [CopulaFamily.GAUSSIAN, CopulaFamily.STUDENT_T]
            if has_heavy_survival:
                mixed_families.extend([CopulaFamily.CLAYTON, CopulaFamily.GUMBEL])

            models.append(
                VineModelSpec("m5_cvine_mixed", VineStructure.C_VINE, mixed_families)
            )

        # D-vine for moderate dimensions
        if n_variables <= 15:
            models.append(
                VineModelSpec("m6_dvine_mixed", VineStructure.D_VINE,
                            [CopulaFamily.GAUSSIAN, CopulaFamily.STUDENT_T])
            )

        # R-vine only for small problems (expensive)
        if n_variables <= 10:
            models.append(
                VineModelSpec("m7_rvine_gaussian", VineStructure.R_VINE, [CopulaFamily.GAUSSIAN])
            )

        # Independence null model (always include as baseline)
        models.append(
            VineModelSpec("m8_independence", VineStructure.INDEPENDENCE, [])
        )

        logger.info(f"Generated {len(models)} models for n={n_variables} variables")
        if constraints is not None:
            logger.info(f" Constraint-aware selection:")
            logger.info(f" - Heavy survival: {has_heavy_survival}")
            logger.info(f" - High correlations: {has_high_correlations}")
            logger.info(f" - Tail dependence: {has_tail_dependence}")

        return models
    
    def phase1_model_selection(
        self,
        constraint_collection: Any,  # ConstraintCollection from constraints.py
        loss_function: Callable,
        n_samples: int
    ) -> List[ModelPosterior]:
        """
        Phase 1: Model selection via short burn-in and BIC computation
        
        Algorithm 1, lines 1-13 of framework
        """
        logger.info(f"Phase 1: Screening {len(self.config.model_specs)} models...")
        
        model_posteriors = []
        
        for idx, model_spec in enumerate(self.config.model_specs):
            logger.info(f"  Training model {idx+1}/{len(self.config.model_specs)}: {model_spec.model_id}")
            
            # Set seed for reproducibility
            seed = self.config.random_seed_base + idx * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Initialize model parameters (placeholder - actual implementation would initialize vine copula)
            # theta_m ~ p(theta | m)
            start_time = datetime.now()
            
            # Short burn-in training
            final_loss = self._train_model_short_burnin(
                model_spec, constraint_collection, loss_function, n_samples
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Compute BIC (Eq. 5.6-5.7)
            k_m = model_spec.count_parameters(len(constraint_collection.variable_names))
            bic = self._compute_bic(final_loss, k_m, n_samples)
            
            logger.info(f"    BIC = {bic:.2f}, k_m = {k_m}, loss = {final_loss:.6f}")
            
            model_posteriors.append(ModelPosterior(
                model_spec=model_spec,
                bic=bic,
                weight=0.0,  # Will be computed next
                n_parameters=k_m,
                final_loss=final_loss,
                converged=True,  # Placeholder
                training_time_seconds=training_time
            ))
        
        # Compute model weights (Eq. 5.10-5.11)
        self._compute_model_weights(model_posteriors)
        
        # Log results
        logger.info("\nPhase 1 Results:")
        logger.info(f"{'Model':<30} {'BIC':>10} {'Weight':>10} {'k_m':>8}")
        logger.info("-" * 60)
        for mp in sorted(model_posteriors, key=lambda x: x.weight, reverse=True):
            logger.info(f"{mp.model_spec.model_id:<30} {mp.bic:>10.2f} {mp.weight:>10.4f} {mp.n_parameters:>8}")
        
        return model_posteriors
    
    def _compute_bic(self, loss: float, k: int, n: int) -> float:
        """
        Compute Bayesian Information Criterion (Eq. 5.7)
        
        BIC_m = -2 * log p(C | theta_m, m) + k_m * log(n)
              = 2 * L(theta_m, m) + k_m * log(n)
        """
        return 2 * loss + k * np.log(n)
    
    def _compute_model_weights(self, model_posteriors: List[ModelPosterior]):
        """
        Compute normalized model weights via BIC approximation (Eq. 5.10-5.11)
        
        Uses numerical stability trick (Eq. 5.12)
        """
        bics = np.array([mp.bic for mp in model_posteriors])
        bic_min = np.min(bics)
        
        # Compute weights with numerical stability
        if self.config.use_uniform_prior:
            # Uniform prior: w_m proportional to exp(-BIC_m / 2)
            log_weights = -(bics - bic_min) / 2.0
        else:
            # Could incorporate non-uniform priors here
            log_weights = -(bics - bic_min) / 2.0
        
        # Normalize
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        # Assign to posteriors
        for mp, weight in zip(model_posteriors, weights):
            mp.weight = float(weight)
    
    def phase2_posterior_sampling(
        self,
        model_posteriors: List[ModelPosterior],
        constraint_collection: Any,
        loss_function: Callable,
        n_samples: int
    ) -> Dict[str, List[Any]]:
        """
        Phase 2: Proportional posterior sampling with full training
        
        Algorithm 1, lines 15-33 of framework
        """
        logger.info(f"\nPhase 2: Proportional posterior sampling...")
        
        # Allocate draws proportionally (line 17)
        draws_allocation = self._allocate_draws(model_posteriors)
        
        logger.info(f"Total draws: {self.config.n_posterior_draws}")
        logger.info(f"Allocation: {draws_allocation}")
        
        posterior_samples = {}
        
        for model_id, n_draws in draws_allocation.items():
            if n_draws == 0:
                continue
            
            logger.info(f"\n  Sampling {n_draws} draws from {model_id}...")
            
            # Find corresponding model spec
            model_spec = next(mp.model_spec for mp in model_posteriors if mp.model_spec.model_id == model_id)
            
            model_samples = []
            
            for j in range(n_draws):
                # Seed for this draw (line 22)
                seed = self.config.random_seed_base + hash(model_id) % 1000 + j
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                logger.info(f"    Draw {j+1}/{n_draws}...")
                
                # Full training (lines 23-29)
                trained_params = self._train_model_full(
                    model_spec, constraint_collection, loss_function, n_samples, draw_id=j
                )
                
                model_samples.append(trained_params)
            
            posterior_samples[model_id] = model_samples
        
        return posterior_samples
    
    def _allocate_draws(self, model_posteriors: List[ModelPosterior]) -> Dict[str, int]:
        """
        Allocate posterior draws proportionally to model weights (line 17)
        
        M_m = ceil(M * w_m)
        
        Prune models with w_m < min_weight (lines 18-20)
        """
        allocation = {}
        
        for mp in model_posteriors:
            if mp.weight < self.config.min_model_weight:
                logger.info(f"  Pruning {mp.model_spec.model_id} (weight={mp.weight:.4f} < {self.config.min_model_weight})")
                allocation[mp.model_spec.model_id] = 0
            else:
                n_draws = int(np.ceil(self.config.n_posterior_draws * mp.weight))
                allocation[mp.model_spec.model_id] = n_draws
        
        return allocation
    
    def _train_model_short_burnin(
        self,
        model_spec: VineModelSpec,
        constraint_collection: Any,
        loss_function: Callable,
        n_samples: int
    ) -> float:
        """
        Short burn-in training for model selection (Phase 1)
        
        Placeholder implementation - actual would train vine copula
        """
        # Simulate training with placeholder loss
        # In actual implementation, this would:
        # 1. Initialize vine copula with specified structure
        # 2. Run gradient descent for E_short epochs
        # 3. Return final loss value
        
        # Placeholder: simulate decreasing loss
        initial_loss = 100.0 + np.random.randn() * 10.0
        
        for epoch in range(self.config.short_burnin_epochs):
            # Simulate one optimization step
            loss_decrease = 0.5 * np.exp(-epoch / 20.0)
            initial_loss -= loss_decrease
        
        # Add model-specific bias (better models = lower loss)
        if model_spec.structure == VineStructure.C_VINE and CopulaFamily.GAUSSIAN in model_spec.families:
            initial_loss *= 0.95  # Slight advantage
        elif model_spec.structure == VineStructure.INDEPENDENCE:
            initial_loss *= 1.20  # Null model worse
        
        return max(initial_loss, 0.1)  # Floor to avoid negative
    
    def _train_model_full(
        self,
        model_spec: VineModelSpec,
        constraint_collection: Any,
        loss_function: Callable,
        n_samples: int,
        draw_id: int
    ) -> Dict[str, Any]:
        """
        Full training for posterior draw (Phase 2)
        
        Placeholder implementation - actual would train vine copula fully
        """
        # In actual implementation:
        # 1. Initialize parameters from prior
        # 2. Run full optimization (E_full epochs)
        # 3. Check convergence
        # 4. Return trained parameters
        
        # Placeholder
        trained_params = {
            "model_spec": model_spec,
            "draw_id": draw_id,
            "converged": True,
            "final_loss": np.random.uniform(0.1, 1.0),
            # Would store: vine parameters, quantile warping functions, etc.
        }
        
        return trained_params
    
    def compute_variance_decomposition(
        self,
        posterior_samples: Dict[str, List[Any]],
        model_posteriors: List[ModelPosterior],
        downstream_estimates: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[str, float]:
        """
        Compute variance decomposition for BMA (Eq. 5.13-5.16)
        
        Args:
            posterior_samples: Dict mapping model_id to list of parameter draws
            model_posteriors: List of ModelPosterior objects with weights
            downstream_estimates: Dict mapping model_id to list of (estimate, variance) tuples
                                 from downstream analyses on synthetic data
        
        Returns:
            Dict with variance components: W_bar, B_within, B_between, T_BMA
        """
        # Extract weights
        weights = {mp.model_spec.model_id: mp.weight for mp in model_posteriors}
        
        # Average within-draw variance (Eq. 5.14)
        W_bar = 0.0
        for model_id, estimates in downstream_estimates.items():
            if model_id not in weights or weights[model_id] == 0:
                continue
            within_draw_vars = [var for _, var in estimates]
            W_m = np.mean(within_draw_vars)
            W_bar += weights[model_id] * W_m
        
        # Within-model between-draw variance (Eq. 5.15)
        B_within = 0.0
        model_means = {}
        for model_id, estimates in downstream_estimates.items():
            if model_id not in weights or weights[model_id] == 0:
                continue
            point_estimates = [est for est, _ in estimates]
            mean_m = np.mean(point_estimates)
            model_means[model_id] = mean_m
            
            M_m = len(point_estimates)
            if M_m > 1:
                B_m = np.var(point_estimates, ddof=1)
                B_within += weights[model_id] * B_m
        
        # BMA pooled estimate
        theta_BMA = sum(weights[model_id] * model_means[model_id] 
                       for model_id in model_means.keys())
        
        # Between-model variance (Eq. 5.16)
        B_between = sum(weights[model_id] * (model_means[model_id] - theta_BMA)**2
                       for model_id in model_means.keys())
        
        # Total variance (Eq. 5.17)
        M = self.config.n_posterior_draws
        T_BMA = W_bar + (1 + 1/M) * B_within + B_between
        
        # Structural uncertainty proportion (Eq. 5.19)
        pi_struct = B_between / T_BMA if T_BMA > 0 else 0.0
        
        return {
            "W_bar": W_bar,
            "B_within": B_within,
            "B_between": B_between,
            "T_BMA": T_BMA,
            "pi_struct": pi_struct,
            "theta_BMA": theta_BMA
        }
    
    def validate_model_weights(self, model_posteriors: List[ModelPosterior]) -> Dict[str, Any]:
        """
        Validation checks for model weights (Section 5.5)
        
        Checks:
        1. No single-model dominance (max_weight < 0.6)
        2. Multi-model support (>= 3 models with weight > 0.05)
        3. Non-negligible between-model variance
        """
        weights = [mp.weight for mp in model_posteriors]
        max_weight = max(weights)
        n_effective = sum(1 for w in weights if w > 0.05)
        
        checks = {
            "max_weight": max_weight,
            "n_effective_models": n_effective,
            "passes": {
                "no_dominance": max_weight < 0.6,
                "multi_model_support": n_effective >= self.config.min_effective_models,
            }
        }
        
        # Interpretation
        if max_weight > 0.8:
            checks["interpretation"] = "Single model dominates - BMA still recommended for honest uncertainty"
        elif not checks["passes"]["multi_model_support"]:
            checks["interpretation"] = "Few effective models - consider expanding model space"
        else:
            checks["interpretation"] = "Good multi-model support"
        
        return checks

    def compute_convergence_diagnostics(
        self,
        loss_history: List[float],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Compute convergence diagnostics for training

        Args:
            loss_history: List of loss values over epochs
            window_size: Window for moving average

        Returns:
            Dict with convergence metrics
        """
        if len(loss_history) < window_size:
            return {'converged': False, 'reason': 'Insufficient history'}

        history = np.array(loss_history)

        # Moving average
        ma = np.convolve(history, np.ones(window_size)/window_size, mode='valid')

        # Relative change in recent epochs
        recent_change = abs(ma[-1] - ma[-window_size]) / (abs(ma[-window_size]) + 1e-8)

        # Trend (should be decreasing)
        recent_trend = np.polyfit(np.arange(len(ma[-window_size:])), ma[-window_size:], 1)[0]

        # Variance in recent epochs (should be small)
        recent_variance = np.var(history[-window_size:])

        # Convergence criteria
        converged = (
            recent_change < self.config.convergence_tolerance and
            recent_trend <= 0 and  # Not increasing
            recent_variance < 0.01 * abs(ma[-1])  # Low variance
        )

        return {
            'converged': converged,
            'recent_change': recent_change,
            'recent_trend': recent_trend,
            'recent_variance': recent_variance,
            'final_loss': history[-1],
            'best_loss': np.min(history),
            'epochs_since_improvement': len(history) - np.argmin(history) - 1
        }

    def run_full_bma(
        self,
        constraint_collection: Any,
        loss_function: Callable,
        n_samples: int
    ) -> BMAResults:
        """
        Run complete BMA algorithm (both phases)
        
        Main entry point for users
        """
        start_time = datetime.now()
        
        # Phase 1: Model selection
        model_posteriors = self.phase1_model_selection(
            constraint_collection, loss_function, n_samples
        )
        
        # Validate weights
        if self.config.validate_model_weights:
            validation = self.validate_model_weights(model_posteriors)
            logger.info(f"\nModel weight validation: {validation}")
        
        # Phase 2: Posterior sampling
        posterior_samples = self.phase2_posterior_sampling(
            model_posteriors, constraint_collection, loss_function, n_samples
        )
        
        # Compute draws per model
        draws_per_model = {k: len(v) for k, v in posterior_samples.items()}
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        results = BMAResults(
            model_posteriors=model_posteriors,
            total_draws=sum(draws_per_model.values()),
            draws_per_model=draws_per_model,
            training_time_total_seconds=total_time
        )
        
        self.results = results
        
        logger.info(f"\nBMA Training Complete!")
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Total draws: {results.total_draws}")
        logger.info(f"\nModel weight summary:")
        for mp in sorted(model_posteriors, key=lambda x: x.weight, reverse=True):
            logger.info(f"  {mp.model_spec.model_id}: {mp.weight:.4f} ({draws_per_model.get(mp.model_spec.model_id, 0)} draws)")
        
        return results
    
    def save_results(self, filepath: Path):
        """Save BMA results to JSON"""
        if self.results is None:
            raise ValueError("No results to save - run BMA first")
        
        results_dict = {
            "model_posteriors": [
                {
                    "model_id": mp.model_spec.model_id,
                    "structure": mp.model_spec.structure,
                    "families": [f.value for f in mp.model_spec.families],
                    "bic": mp.bic,
                    "weight": mp.weight,
                    "n_parameters": mp.n_parameters,
                    "final_loss": mp.final_loss
                }
                for mp in self.results.model_posteriors
            ],
            "summary": self.results.summary(),
            "config": {
                "n_posterior_draws": self.config.n_posterior_draws,
                "short_burnin_epochs": self.config.short_burnin_epochs,
                "full_training_epochs": self.config.full_training_epochs
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
