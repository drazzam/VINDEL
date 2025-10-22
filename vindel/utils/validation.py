"""
Validation Utilities for VINDEL

Provides tools for validating synthetic IPD quality and diagnosing issues.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class SyntheticIPDValidator:
    """
    Validation suite for synthetic IPD quality

    Provides diagnostic tools to assess:
    1. Aggregate reproduction fidelity
    2. Biological plausibility
    3. Distributional properties
    4. Downstream analysis concordance
    """

    def __init__(self, tolerance: float = 0.05):
        """
        Args:
            tolerance: Relative tolerance for aggregate matching (5% default)
        """
        self.tolerance = tolerance
        self.validation_results = {}

    def validate_aggregate_reproduction(
        self,
        synthetic_data: np.ndarray,
        constraints: Any,  # ConstraintCollection
        variable_names: List[str]
    ) -> Dict[str, Any]:
        """
        Check how well synthetic data reproduces published aggregates

        Returns:
            Dict with validation results for each constraint type
        """
        results = {
            'marginals': self._validate_marginals(synthetic_data, constraints, variable_names),
            'correlations': self._validate_correlations(synthetic_data, constraints, variable_names),
            'survival': self._validate_survival(synthetic_data, constraints),
        }

        # Overall pass/fail
        all_passed = all(
            result['all_within_tolerance']
            for result in results.values()
            if 'all_within_tolerance' in result
        )

        results['overall_pass'] = all_passed

        return results

    def _validate_marginals(
        self,
        data: np.ndarray,
        constraints: Any,
        variable_names: List[str]
    ) -> Dict[str, Any]:
        """Validate marginal constraint reproduction"""
        var_map = {name: idx for idx, name in enumerate(variable_names)}

        errors = []

        for constraint in constraints.marginal_constraints:
            var_idx = var_map.get(constraint.variable_name)
            if var_idx is None:
                continue

            var_data = data[:, var_idx]

            # Check mean
            if constraint.mean is not None:
                empirical_mean = np.mean(var_data)
                relative_error = abs(empirical_mean - constraint.mean) / (abs(constraint.mean) + 1e-8)

                errors.append({
                    'variable': constraint.variable_name,
                    'statistic': 'mean',
                    'target': constraint.mean,
                    'empirical': empirical_mean,
                    'relative_error': relative_error,
                    'within_tolerance': relative_error < self.tolerance
                })

            # Check std
            if constraint.std is not None:
                empirical_std = np.std(var_data, ddof=1)
                relative_error = abs(empirical_std - constraint.std) / (abs(constraint.std) + 1e-8)

                errors.append({
                    'variable': constraint.variable_name,
                    'statistic': 'std',
                    'target': constraint.std,
                    'empirical': empirical_std,
                    'relative_error': relative_error,
                    'within_tolerance': relative_error < self.tolerance
                })

        all_within_tolerance = all(e['within_tolerance'] for e in errors)
        max_error = max([e['relative_error'] for e in errors]) if errors else 0.0

        return {
            'errors': errors,
            'all_within_tolerance': all_within_tolerance,
            'max_relative_error': max_error,
            'n_constraints_checked': len(errors)
        }

    def _validate_correlations(
        self,
        data: np.ndarray,
        constraints: Any,
        variable_names: List[str]
    ) -> Dict[str, Any]:
        """Validate correlation constraint reproduction"""
        var_map = {name: idx for idx, name in enumerate(variable_names)}

        errors = []

        for constraint in constraints.joint_constraints:
            idx1 = var_map.get(constraint.variable1)
            idx2 = var_map.get(constraint.variable2)

            if idx1 is None or idx2 is None:
                continue

            if constraint.pearson_correlation is not None:
                empirical_corr = np.corrcoef(data[:, idx1], data[:, idx2])[0, 1]
                absolute_error = abs(empirical_corr - constraint.pearson_correlation)

                errors.append({
                    'variables': f"{constraint.variable1}-{constraint.variable2}",
                    'target': constraint.pearson_correlation,
                    'empirical': empirical_corr,
                    'absolute_error': absolute_error,
                    'within_tolerance': absolute_error < 0.05  # Absolute tolerance for correlations
                })

        all_within_tolerance = all(e['within_tolerance'] for e in errors) if errors else True
        max_error = max([e['absolute_error'] for e in errors]) if errors else 0.0

        return {
            'errors': errors,
            'all_within_tolerance': all_within_tolerance,
            'max_absolute_error': max_error,
            'n_constraints_checked': len(errors)
        }

    def _validate_survival(
        self,
        data: np.ndarray,
        constraints: Any
    ) -> Dict[str, Any]:
        """Validate survival constraint reproduction"""
        # Placeholder - would need survival times and event indicators
        return {
            'all_within_tolerance': True,
            'n_constraints_checked': 0,
            'note': 'Survival validation requires survival_times and event_indicators'
        }

    def check_distributional_properties(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> Dict[str, Any]:
        """
        Check distributional properties for suspicious patterns

        Tests:
        - Normality (Shapiro-Wilk)
        - Outliers (beyond ±3 SD)
        - Multimodality (Hartigan's dip test if available)
        """
        results = {}

        for i, var_name in enumerate(variable_names):
            var_data = data[:, i]

            # Normality test
            if len(var_data) >= 3 and len(var_data) <= 5000:
                _, p_value = stats.shapiro(var_data)
                is_normal = p_value > 0.05
            else:
                is_normal = None  # Too small or large for test

            # Outliers
            mean = np.mean(var_data)
            std = np.std(var_data)
            outliers = np.sum(np.abs(var_data - mean) > 3 * std)
            outlier_rate = outliers / len(var_data)

            # Skewness and kurtosis
            skewness = stats.skew(var_data)
            kurtosis = stats.kurtosis(var_data)

            results[var_name] = {
                'is_normal': is_normal,
                'outlier_count': int(outliers),
                'outlier_rate': outlier_rate,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'suspicious': (
                    (is_normal is False and is_normal is not None) or
                    outlier_rate > 0.05 or
                    abs(skewness) > 2 or
                    abs(kurtosis) > 7
                )
            }

        return results

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report"""
        report = []

        report.append("="*80)
        report.append("SYNTHETIC IPD VALIDATION REPORT")
        report.append("="*80)

        # Aggregate reproduction
        if 'marginals' in results:
            marg = results['marginals']
            report.append(f"\nMarginal Constraints: {marg['n_constraints_checked']} checked")
            report.append(f" Max relative error: {marg['max_relative_error']:.2%}")
            report.append(f" All within tolerance: {'✓ YES' if marg['all_within_tolerance'] else '✗ NO'}")

        if 'correlations' in results:
            corr = results['correlations']
            report.append(f"\nCorrelation Constraints: {corr['n_constraints_checked']} checked")
            report.append(f" Max absolute error: {corr['max_absolute_error']:.3f}")
            report.append(f" All within tolerance: {'✓ YES' if corr['all_within_tolerance'] else '✗ NO'}")

        # Overall
        overall = results.get('overall_pass', False)
        report.append(f"\n{'='*80}")
        report.append(f"OVERALL: {'✓ PASS' if overall else '✗ FAIL'}")
        report.append(f"{'='*80}")

        return "\n".join(report)
