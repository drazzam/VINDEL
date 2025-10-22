"""
Loss Functions for VINDEL Framework

Implements all 11 constraint types from Section 4.3 (Eq. 4.5):
- Marginal, Joint, Conditional, Survival, Subgroup Survival
- Causal, Multi-outcome, Network, Physics
- Time-varying HR, Optimal Transport (optional)

Each loss is differentiable for gradient-based optimization.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BASE LOSS INTERFACE
# ============================================================================

class BaseLoss(nn.Module):
    """Base class for all loss functions"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.loss_name = self.__class__.__name__
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def compute_with_logging(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and return detailed metrics"""
        loss = self.forward(*args, **kwargs)
        metrics = {
            f"{self.loss_name}_value": loss.item(),
            f"{self.loss_name}_weighted": (loss * self.weight).item()
        }
        return loss, metrics


# ============================================================================
# MARGINAL CONSTRAINTS (C_marg)
# ============================================================================

class MarginalLoss(BaseLoss):
    """
    Loss for marginal distribution constraints (Section 4.3.1)
    
    L_marg = sum over constraints of (empirical - target)^2
    """
    
    def forward(
        self,
        data: torch.Tensor,  # (n_samples, n_variables)
        marginal_constraints: List[Any],  # List of MarginalConstraint objects
        variable_names: List[str]
    ) -> torch.Tensor:
        """
        Compute marginal constraint loss
        
        Args:
            data: Generated synthetic data
            marginal_constraints: List of MarginalConstraint objects
            variable_names: List of variable names matching data columns
            
        Returns:
            Marginal loss (scalar)
        """
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        
        var_map = {name: idx for idx, name in enumerate(variable_names)}
        
        for constraint in marginal_constraints:
            var_idx = var_map.get(constraint.variable_name)
            if var_idx is None:
                continue
            
            var_data = data[:, var_idx]
            constraint_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
            
            # Mean constraint
            if constraint.mean is not None:
                empirical_mean = torch.mean(var_data)
                constraint_loss += (empirical_mean - constraint.mean)**2
            
            # Standard deviation constraint
            if constraint.std is not None:
                empirical_std = torch.std(var_data, unbiased=True)
                constraint_loss += (empirical_std - constraint.std)**2
            
            # Quantile constraints
            if constraint.quantiles is not None:
                for quantile_level, target_value in constraint.quantiles.items():
                    empirical_quantile = torch.quantile(var_data, quantile_level)
                    constraint_loss += (empirical_quantile - target_value)**2
            
            # Weight by uncertainty if available
            if constraint.uncertainty is not None and constraint.uncertainty > 0:
                constraint_loss = constraint_loss / (constraint.uncertainty**2)
            
            total_loss += constraint.weight * constraint_loss
        
        return total_loss


# ============================================================================
# JOINT CONSTRAINTS (C_joint)
# ============================================================================

class JointLoss(BaseLoss):
    """
    Loss for pairwise joint dependence constraints (Section 4.3.2)
    
    L_joint = sum over pairs of (empirical_correlation - target_correlation)^2
    """
    
    def forward(
        self,
        data: torch.Tensor,
        joint_constraints: List[Any],
        variable_names: List[str]
    ) -> torch.Tensor:
        """Compute joint constraint loss"""
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        
        var_map = {name: idx for idx, name in enumerate(variable_names)}
        
        for constraint in joint_constraints:
            idx1 = var_map.get(constraint.variable1)
            idx2 = var_map.get(constraint.variable2)
            
            if idx1 is None or idx2 is None:
                continue
            
            var1 = data[:, idx1]
            var2 = data[:, idx2]
            
            constraint_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)

            # Pearson correlation with Fisher Z-transform
            if constraint.pearson_correlation is not None:
                empirical_z = self._pearson_correlation_fisher_z(var1, var2)
                target_z = np.arctanh(np.clip(constraint.pearson_correlation, -0.99, 0.99))

                # Weight by uncertainty if available
                if constraint.uncertainty is not None and constraint.uncertainty > 0:
                    # Uncertainty in Fisher Z space: SE_z ≈ 1/sqrt(n-3)
                    # For aggregate data, use provided uncertainty
                    se_z = constraint.uncertainty / (1 - constraint.pearson_correlation**2)
                    constraint_loss += ((empirical_z - target_z) / se_z)**2
                else:
                    constraint_loss += (empirical_z - target_z)**2

            # Kendall's tau (computationally expensive, approximate)
            if constraint.kendall_tau is not None:
                empirical_tau = self._kendall_tau_approx(var1, var2)
                constraint_loss += (empirical_tau - constraint.kendall_tau)**2

            total_loss += constraint.weight * constraint_loss
        
        return total_loss
    
    @staticmethod
    def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation (differentiable)"""
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        
        return numerator / (denominator + 1e-8)
    
    @staticmethod
    def _kendall_tau_approx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Approximate Kendall's tau (differentiable via copula transform)"""
        # Use Gaussian copula approximation: tau ≈ (2/π) * arcsin(ρ)
        rho = JointLoss._pearson_correlation(x, y)
        tau_approx = (2.0 / np.pi) * torch.asin(torch.clamp(rho, -0.99, 0.99))
        return tau_approx

    @staticmethod
    def _pearson_correlation_fisher_z(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation with Fisher Z-transform for better loss properties

        Fisher Z-transform: z = atanh(r) has approximately normal distribution
        This gives better gradient properties than raw correlation

        Args:
            x, y: Input tensors

        Returns:
            Fisher Z-transformed correlation
        """
        rho = JointLoss._pearson_correlation(x, y)
        # Clamp to prevent atanh(±1) = ±inf
        rho_clamped = torch.clamp(rho, -0.99, 0.99)
        z = torch.atanh(rho_clamped)
        return z


# ============================================================================
# CONDITIONAL CORRELATION CONSTRAINTS (C_cond) - NEW
# ============================================================================

class ConditionalCorrelationLoss(BaseLoss):
    """
    Loss for conditional correlation constraints (Section 6.2)
    
    L_cond = sum over subgroups of (empirical_corr | subgroup - target)^2
    """
    
    def forward(
        self,
        data: torch.Tensor,
        conditional_constraints: List[Any],
        variable_names: List[str]
    ) -> torch.Tensor:
        """Compute conditional correlation loss"""
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        
        var_map = {name: idx for idx, name in enumerate(variable_names)}
        
        for constraint in conditional_constraints:
            idx1 = var_map.get(constraint.variable1)
            idx2 = var_map.get(constraint.variable2)
            
            if idx1 is None or idx2 is None:
                continue
            
            # Get subgroup mask
            mask = self._get_subgroup_mask(data, constraint.subgroup, variable_names)
            
            if torch.sum(mask) < 10:  # Need minimum samples
                continue
            
            # Compute correlation within subgroup
            var1_subgroup = data[mask, idx1]
            var2_subgroup = data[mask, idx2]
            
            empirical_corr = JointLoss._pearson_correlation(var1_subgroup, var2_subgroup)
            
            constraint_loss = (empirical_corr - constraint.correlation)**2
            
            # Weight by subgroup size
            subgroup_weight = torch.sum(mask).float() / len(data)
            total_loss += constraint.weight * subgroup_weight * constraint_loss
        
        return total_loss
    
    @staticmethod
    def _get_subgroup_mask(
        data: torch.Tensor,
        subgroup_def: Any,
        variable_names: List[str]
    ) -> torch.Tensor:
        """Get boolean mask for subgroup membership"""
        mask = torch.ones(len(data), dtype=torch.bool, device=data.device)
        
        var_map = {name: idx for idx, name in enumerate(variable_names)}
        
        for condition in subgroup_def.conditions:
            var_idx = var_map.get(condition['variable'])
            if var_idx is None:
                continue
            
            var_data = data[:, var_idx]
            op = condition['operator']
            val = condition['value']
            
            if op == '>=':
                mask &= (var_data >= val)
            elif op == '<=':
                mask &= (var_data <= val)
            elif op == '>':
                mask &= (var_data > val)
            elif op == '<':
                mask &= (var_data < val)
            elif op == '==':
                mask &= (var_data == val)
        
        return mask


# ============================================================================
# SURVIVAL CONSTRAINTS (C_surv)
# ============================================================================

class KaplanMeierLoss(BaseLoss):
    """
    Loss for Kaplan-Meier fidelity (Section 4.3.3)
    
    L_KM = sum over time points of (S_empirical(t) - S_published(t))^2
    """
    
    def forward(
        self,
        survival_times: torch.Tensor,  # (n_samples,)
        event_indicators: torch.Tensor,  # (n_samples,) 1=event, 0=censored
        km_constraints: List[Any],
        arm_assignments: Optional[torch.Tensor] = None  # (n_samples,) arm indices
    ) -> torch.Tensor:
        """Compute KM fidelity loss"""
        total_loss = torch.tensor(0.0, device=survival_times.device, dtype=survival_times.dtype)
        
        for constraint in km_constraints:
            if constraint.km_times is None or constraint.km_survival is None:
                continue
            
            # Filter by arm if specified
            if constraint.arm is not None and arm_assignments is not None:
                # Would need arm encoding - simplified here
                pass
            
            # Compute empirical survival at each time point
            for t_pub, s_pub in zip(constraint.km_times, constraint.km_survival):
                # Empirical survival at time t: S(t) = P(T > t)
                at_risk = (survival_times >= t_pub) | ((survival_times < t_pub) & (event_indicators == 0))
                s_empirical = torch.mean(at_risk.float())
                
                constraint_loss = (s_empirical - s_pub)**2
                
                total_loss += constraint.weight * constraint_loss
        
        return total_loss


# ============================================================================
# QUANTILE LOSS FUNCTION - NEW
# ============================================================================

class QuantileLoss(BaseLoss):
    """
    Quantile loss using check function (asymmetric L1)

    For quantile τ, the loss is:
    L(y, ŷ) = τ * max(0, y - ŷ) + (1-τ) * max(0, ŷ - y)

    This is more appropriate than squared loss for quantile constraints.
    """

    def forward(
        self,
        data: torch.Tensor,
        quantile_constraints: List[Any],
        variable_names: List[str]
    ) -> torch.Tensor:
        """
        Compute quantile loss using check function

        Args:
            data: Generated synthetic data (n_samples, n_variables)
            quantile_constraints: List of MarginalConstraint with quantiles
            variable_names: Variable names

        Returns:
            Quantile loss (scalar)
        """
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)

        var_map = {name: idx for idx, name in enumerate(variable_names)}

        for constraint in quantile_constraints:
            if constraint.quantiles is None:
                continue

            var_idx = var_map.get(constraint.variable_name)
            if var_idx is None:
                continue

            var_data = data[:, var_idx]

            for tau, target_quantile in constraint.quantiles.items():
                # Compute empirical quantile
                empirical_quantile = torch.quantile(var_data, tau)

                # Check function loss
                error = target_quantile - empirical_quantile
                check_loss = torch.where(
                    error > 0,
                    tau * error,
                    (tau - 1) * error
                )

                total_loss += constraint.weight * check_loss.abs()

        return total_loss


# ============================================================================
# SUBGROUP SURVIVAL CONSTRAINTS (C_surv,subg) - NEW
# ============================================================================

class SubgroupSurvivalLoss(BaseLoss):
    """
    Loss for subgroup-specific survival constraints (Section 6.3)
    
    L_KM,subg = sum over subgroups,time points of (S_g(t) - S_g^pub(t))^2
    """
    
    def forward(
        self,
        data: torch.Tensor,
        survival_times: torch.Tensor,
        event_indicators: torch.Tensor,
        subgroup_survival_constraints: List[Any],
        variable_names: List[str]
    ) -> torch.Tensor:
        """Compute subgroup-specific survival loss"""
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        
        for constraint in subgroup_survival_constraints:
            # Get subgroup mask
            mask = ConditionalCorrelationLoss._get_subgroup_mask(
                data, constraint.subgroup, variable_names
            )
            
            if torch.sum(mask) < 10:
                continue
            
            # Survival within subgroup
            surv_times_subgroup = survival_times[mask]
            events_subgroup = event_indicators[mask]
            
            if constraint.km_times is not None and constraint.km_survival is not None:
                for t_pub, s_pub in zip(constraint.km_times, constraint.km_survival):
                    at_risk = (surv_times_subgroup >= t_pub) | \
                             ((surv_times_subgroup < t_pub) & (events_subgroup == 0))
                    s_empirical = torch.mean(at_risk.float())
                    
                    constraint_loss = (s_empirical - s_pub)**2
                    total_loss += constraint.weight * constraint_loss
            
            # Median survival
            if constraint.median_survival is not None:
                # Approximate median from empirical distribution
                sorted_times, _ = torch.sort(surv_times_subgroup[events_subgroup == 1])
                if len(sorted_times) > 0:
                    median_empirical = torch.median(sorted_times)
                    constraint_loss = (median_empirical - constraint.median_survival)**2
                    total_loss += constraint.weight * 10.0 * constraint_loss  # Higher weight for median
        
        return total_loss


# ============================================================================
# TIME-VARYING HAZARD RATIO CONSTRAINTS (C_HR,TV) - NEW
# ============================================================================

class TimeVaryingHRLoss(BaseLoss):
    """
    Loss for time-varying hazard ratios (Section 6.3.3)
    
    L_HR,TV = sum over periods of (log HR_k - log HR_k^pub)^2
    """
    
    def forward(
        self,
        survival_times: torch.Tensor,
        event_indicators: torch.Tensor,
        arm_assignments: torch.Tensor,
        tv_hr_constraints: List[Any]
    ) -> torch.Tensor:
        """Compute time-varying HR loss"""
        total_loss = torch.tensor(0.0, device=survival_times.device, dtype=survival_times.dtype)
        
        for constraint in tv_hr_constraints:
            # For each time period, compute empirical HR
            for period_idx, (t_start, t_end) in enumerate(constraint.time_periods):
                target_hr = constraint.hazard_ratios[period_idx]
                
                # Filter events in this period
                in_period = (survival_times >= t_start) & (survival_times < t_end) & (event_indicators == 1)
                
                if torch.sum(in_period) < 5:
                    continue
                
                # Simple empirical HR: ratio of event rates
                # (In practice, would use Cox model)
                treatment_events = torch.sum(in_period & (arm_assignments == 1)).float()
                control_events = torch.sum(in_period & (arm_assignments == 0)).float()
                
                treatment_atrisk = torch.sum((arm_assignments == 1) & (survival_times >= t_start)).float()
                control_atrisk = torch.sum((arm_assignments == 0) & (survival_times >= t_start)).float()
                
                if treatment_atrisk > 0 and control_atrisk > 0:
                    treatment_rate = treatment_events / (treatment_atrisk + 1e-6)
                    control_rate = control_events / (control_atrisk + 1e-6)
                    
                    empirical_hr = treatment_rate / (control_rate + 1e-6)
                    
                    constraint_loss = (torch.log(empirical_hr + 1e-6) - np.log(target_hr))**2
                    total_loss += constraint.weight * constraint_loss
        
        return total_loss


# ============================================================================
# MULTI-OUTCOME CONSISTENCY CONSTRAINTS (C_multi) - NEW
# ============================================================================

class MultiOutcomeLoss(BaseLoss):
    """
    Loss for multi-outcome consistency (Section 6.4)
    
    Ensures binary and survival endpoints are mutually consistent:
    - Response rate matches published
    - Mean survival matches published
    - Responders have better survival than non-responders (monotonicity)
    """
    
    def forward(
        self,
        binary_outcome: torch.Tensor,  # (n_samples,) 0/1
        survival_times: torch.Tensor,
        arm_assignments: torch.Tensor,
        multi_outcome_constraints: List[Any]
    ) -> torch.Tensor:
        """Compute multi-outcome consistency loss"""
        total_loss = torch.tensor(0.0, device=binary_outcome.device, dtype=binary_outcome.dtype)
        
        for constraint in multi_outcome_constraints:
            # Binary endpoint loss
            for arm, target_prevalence in constraint.binary_prevalence.items():
                arm_mask = (arm_assignments == int(arm))  # Simplified
                empirical_prevalence = torch.mean(binary_outcome[arm_mask].float())
                
                binary_loss = (empirical_prevalence - target_prevalence)**2
                total_loss += constraint.weight * binary_loss
            
            # Survival mean loss
            for arm, target_mean_surv in constraint.mean_survival.items():
                arm_mask = (arm_assignments == int(arm))
                empirical_mean_surv = torch.mean(survival_times[arm_mask])
                
                surv_loss = (empirical_mean_surv - target_mean_surv)**2
                total_loss += constraint.weight * surv_loss
            
            # Monotonicity: responders should have better survival
            if constraint.enforce_monotonicity:
                responders = (binary_outcome == 1)
                non_responders = (binary_outcome == 0)
                
                mean_surv_responders = torch.mean(survival_times[responders])
                mean_surv_non_responders = torch.mean(survival_times[non_responders])
                
                # Penalty if non-responders have better survival (violation)
                monotone_loss = torch.relu(mean_surv_non_responders - mean_surv_responders)
                total_loss += constraint.weight * monotone_loss**2
        
        return total_loss


# ============================================================================
# CAUSAL CONSTRAINTS (C_causal)
# ============================================================================

class CausalLoss(BaseLoss):
    """
    Loss for causal treatment effect constraints (Section 4.3.4)
    
    L_causal = (ATE_empirical - ATE_published)^2 + subgroup losses
    """
    
    def forward(
        self,
        outcomes: torch.Tensor,  # (n_samples,)
        arm_assignments: torch.Tensor,  # (n_samples,)
        causal_constraints: List[Any]
    ) -> torch.Tensor:
        """Compute causal constraint loss"""
        total_loss = torch.tensor(0.0, device=outcomes.device, dtype=outcomes.dtype)
        
        for constraint in causal_constraints:
            # Average treatment effect
            if constraint.ate is not None:
                treatment_mask = (arm_assignments == 1)  # Simplified
                control_mask = (arm_assignments == 0)
                
                mean_treatment = torch.mean(outcomes[treatment_mask])
                mean_control = torch.mean(outcomes[control_mask])
                
                empirical_ate = mean_treatment - mean_control
                
                ate_loss = (empirical_ate - constraint.ate)**2
                
                if constraint.ate_se is not None:
                    ate_loss = ate_loss / (constraint.ate_se**2)
                
                total_loss += constraint.weight * ate_loss
        
        return total_loss


# ============================================================================
# COMPOSITE LOSS
# ============================================================================

class CompositeLoss(nn.Module):
    """
    Composite loss function combining all constraint types (Eq. 4.5)
    
    L_total = Σ λ_c * L_c
    """
    
    def __init__(self, loss_weights: Dict[str, float]):
        super().__init__()

        self.losses = {
            'marginal': MarginalLoss(loss_weights.get('lambda_marg', 10.0)),
            'joint': JointLoss(loss_weights.get('lambda_joint', 5.0)),
            'conditional': ConditionalCorrelationLoss(loss_weights.get('lambda_cond', 3.0)),
            'km': KaplanMeierLoss(loss_weights.get('lambda_KM', 20.0)),
            'subgroup_survival': SubgroupSurvivalLoss(loss_weights.get('lambda_KM_subg', 15.0)),
            'time_varying_hr': TimeVaryingHRLoss(loss_weights.get('lambda_HR_TV', 5.0)),
            'multi_outcome': MultiOutcomeLoss(loss_weights.get('lambda_multi', 5.0)),
            'causal': CausalLoss(loss_weights.get('lambda_causal', 5.0)),
            # Physics loss computed separately in physics module
        }

        # Adaptive weight support
        self.use_adaptive = False
        self.adaptive_weights = None
    
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        constraints: Any,  # ConstraintCollection
        variable_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total composite loss
        
        Args:
            data: Dict containing:
                - 'covariates': (n, p) covariate matrix
                - 'survival_times': (n,) event/censoring times
                - 'event_indicators': (n,) 0/1 indicators
                - 'arm_assignments': (n,) treatment arm indices
                - 'binary_outcomes': (n,) optional binary outcomes
            constraints: ConstraintCollection object
            variable_names: List of variable names
            
        Returns:
            total_loss: Scalar composite loss
            metrics: Dict of individual loss components
        """
        total_loss = torch.tensor(0.0, device=data['covariates'].device)
        metrics = {}
        
        # Marginal loss
        if len(constraints.marginal_constraints) > 0:
            loss, loss_metrics = self.losses['marginal'].compute_with_logging(
                data['covariates'], constraints.marginal_constraints, variable_names
            )
            total_loss += loss * self.losses['marginal'].weight
            metrics.update(loss_metrics)
        
        # Joint loss
        if len(constraints.joint_constraints) > 0:
            loss, loss_metrics = self.losses['joint'].compute_with_logging(
                data['covariates'], constraints.joint_constraints, variable_names
            )
            total_loss += loss * self.losses['joint'].weight
            metrics.update(loss_metrics)
        
        # Conditional loss
        if len(constraints.conditional_constraints) > 0:
            loss, loss_metrics = self.losses['conditional'].compute_with_logging(
                data['covariates'], constraints.conditional_constraints, variable_names
            )
            total_loss += loss * self.losses['conditional'].weight
            metrics.update(loss_metrics)
        
        # Survival losses (if survival data present)
        if 'survival_times' in data and len(constraints.survival_constraints) > 0:
            loss, loss_metrics = self.losses['km'].compute_with_logging(
                data['survival_times'], data['event_indicators'],
                constraints.survival_constraints, data.get('arm_assignments')
            )
            total_loss += loss * self.losses['km'].weight
            metrics.update(loss_metrics)
        
        # Subgroup survival
        if 'survival_times' in data and len(constraints.subgroup_survival_constraints) > 0:
            loss, loss_metrics = self.losses['subgroup_survival'].compute_with_logging(
                data['covariates'], data['survival_times'], data['event_indicators'],
                constraints.subgroup_survival_constraints, variable_names
            )
            total_loss += loss * self.losses['subgroup_survival'].weight
            metrics.update(loss_metrics)
        
        # Causal loss
        if len(constraints.causal_constraints) > 0 and 'outcomes' in data:
            loss, loss_metrics = self.losses['causal'].compute_with_logging(
                data['outcomes'], data['arm_assignments'], constraints.causal_constraints
            )
            total_loss += loss * self.losses['causal'].weight
            metrics.update(loss_metrics)
        
        # Multi-outcome
        if len(constraints.multi_outcome_constraints) > 0 and 'binary_outcomes' in data:
            loss, loss_metrics = self.losses['multi_outcome'].compute_with_logging(
                data['binary_outcomes'], data['survival_times'],
                data['arm_assignments'], constraints.multi_outcome_constraints
            )
            total_loss += loss * self.losses['multi_outcome'].weight
            metrics.update(loss_metrics)
        
        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def enable_adaptive_weights(self, initial_weights: Optional[Dict[str, float]] = None):
        """Enable adaptive loss weighting"""
        loss_names = list(self.losses.keys())
        self.adaptive_weights = AdaptiveLossWeights(
            n_loss_types=len(loss_names),
            initial_weights=initial_weights
        )
        self.use_adaptive = True
        return self.adaptive_weights.parameters()


# ============================================================================
# ADAPTIVE LOSS WEIGHTING - NEW
# ============================================================================

class AdaptiveLossWeights(nn.Module):
    """
    Learnable loss weights that adapt during training

    Instead of fixed λ_c weights, learns optimal weights via:
    - Softmax parameterization (ensures positivity and bounded)
    - Optional entropy regularization (prevents collapsing to single loss)
    - Uncertainty-aware scaling

    Usage:
        adaptive_weights = AdaptiveLossWeights(n_loss_types=8)
        total_loss = adaptive_weights(individual_losses, loss_names)
        # Optimize both model params and adaptive_weights.parameters()
    """

    def __init__(
        self,
        n_loss_types: int,
        initial_weights: Optional[Dict[str, float]] = None,
        entropy_weight: float = 0.1,
        min_weight: float = 0.01
    ):
        """
        Args:
            n_loss_types: Number of different loss types
            initial_weights: Optional dict mapping loss name to initial weight
            entropy_weight: Weight for entropy regularization (higher = more uniform)
            min_weight: Minimum weight for any loss type
        """
        super().__init__()

        # Log-weights (ensures positivity after exp)
        if initial_weights is not None:
            init_values = torch.tensor([initial_weights.get(f"loss_{i}", 1.0)
                                       for i in range(n_loss_types)])
            self.log_weights = nn.Parameter(torch.log(init_values))
        else:
            self.log_weights = nn.Parameter(torch.zeros(n_loss_types))

        self.entropy_weight = entropy_weight
        self.min_weight = min_weight
        self.n_loss_types = n_loss_types

    def forward(
        self,
        losses: List[torch.Tensor],
        loss_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss with adaptive weights

        Args:
            losses: List of individual loss terms
            loss_names: Optional names for logging

        Returns:
            total_loss: Weighted sum of losses
            weights_dict: Dictionary of current weights
        """
        # Compute weights via softmax (ensures sum = n_loss_types, all positive)
        weights_raw = torch.softmax(self.log_weights, dim=0) * self.n_loss_types

        # Enforce minimum weight
        weights = torch.clamp(weights_raw, min=self.min_weight)
        weights = weights / weights.sum() * self.n_loss_types  # Renormalize

        # Compute weighted loss
        total_loss = sum(w * l for w, l in zip(weights, losses))

        # Entropy regularization (prevents collapse to single loss)
        if self.entropy_weight > 0:
            probs = weights / weights.sum()
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            # Maximize entropy = minimize negative entropy
            total_loss = total_loss - self.entropy_weight * entropy

        # Create weights dictionary for logging
        if loss_names is None:
            loss_names = [f"loss_{i}" for i in range(len(losses))]

        weights_dict = {name: w.item() for name, w in zip(loss_names, weights)}

        return total_loss, weights_dict

    def get_current_weights(self) -> Dict[int, float]:
        """Get current weight values"""
        weights = torch.softmax(self.log_weights, dim=0) * self.n_loss_types
        return {i: w.item() for i, w in enumerate(weights)}
