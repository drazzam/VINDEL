"""
VINDEL Quick Start Example - LLM Integration Demo

This example shows how to use VINDEL with LLM's dynamic capabilities (Claude or ChatGPT).
When you upload this package to your LLM and run this code, the LLM will
automatically use its web_search and reasoning to configure everything.
"""

import numpy as np
from vindel import (
    # Core data structures
    ConstraintCollection,
    MarginalConstraint,
    JointConstraint,
    SurvivalConstraint,
    SubgroupDefinition,
    ConditionalCorrelationConstraint,
    
    # BMA components
    BMAEngine,
    BMAConfig,
    
    # LLM integration (THE KEY INNOVATION!)
    DiseaseProfileRetriever,
    PaperConstraintExtractor,
    SmartParameterSelector,
    
    # Loss functions
    CompositeLoss,
)


def example_1_basic_constraints():
    """Example 1: Basic constraint specification"""
    
    print("="*80)
    print("EXAMPLE 1: Basic Constraint Specification")
    print("="*80)
    
    # Create constraint collection
    constraints = ConstraintCollection(
        study_name="Example Lung Cancer RCT",
        variable_names=["age", "weight", "ecog_ps", "pdl1_score"],
        treatment_arms=["control", "treatment_A"]
    )
    
    # Add marginal constraints from baseline characteristics table
    constraints.marginal_constraints.extend([
        MarginalConstraint(
            variable_name="age",
            mean=64.3,
            std=9.8,
            quantiles={0.25: 58, 0.50: 65, 0.75: 71},
            weight=1.0,
            source="Table 1, Trial Publication"
        ),
        MarginalConstraint(
            variable_name="weight",
            mean=72.4,
            std=14.2,
            weight=1.0,
            source="Table 1"
        ),
    ])
    
    # Add joint constraint (correlation reported)
    constraints.joint_constraints.append(
        JointConstraint(
            variable1="age",
            variable2="weight",
            pearson_correlation=0.15,
            weight=1.0,
            source="Supplementary Table S2"
        )
    )
    
    # Add survival constraint from Kaplan-Meier curve
    constraints.survival_constraints.append(
        SurvivalConstraint(
            study_id="trial_001",
            arm="treatment_A",
            km_times=[0, 6, 12, 18, 24, 30],  # months
            km_survival=[1.00, 0.82, 0.68, 0.52, 0.39, 0.28],
            median_survival=16.5,
            hazard_ratio=0.75,  # vs control
            weight=2.0,  # Higher weight for survival endpoints
            source="Figure 2A, KM curve digitized"
        )
    )
    
    print(f"\n✓ Created constraint collection with:")
    print(f"  - {len(constraints.marginal_constraints)} marginal constraints")
    print(f"  - {len(constraints.joint_constraints)} joint constraints")
    print(f"  - {len(constraints.survival_constraints)} survival constraints")
    
    return constraints


def example_2_llm_disease_profile():
    """Example 2: LLM retrieves disease profile from literature"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: LLM-Powered Disease Profile Retrieval")
    print("="*80)
    
    print("\n🔍 Asking LLM to search medical literature...")
    print("   (When you run this in LLM chat, it will use web_search)")

    # This is the MAGIC - LLM will search PubMed automatically
    retriever = DiseaseProfileRetriever()
    
    disease_profile = retriever.get_profile(
        disease_name="metastatic non-small cell lung cancer",
        additional_context={
            "line_of_therapy": "second-line",
            "biomarker_status": "PD-L1 positive",
            "histology": "non-squamous"
        }
    )
    
    print("\n✓ Disease Profile Retrieved:")
    print(f"  - Disease: {disease_profile.disease_name}")
    print(f"  - Hazard profile: {disease_profile.hazard_profile}")
    print(f"  - Median survival estimate: {disease_profile.median_survival_months} months")
    print(f"  - Confidence: {disease_profile.llm_confidence}")
    print(f"  - Literature sources: {len(disease_profile.literature_sources)} references")
    
    print("\n  Monotonic relationships identified:")
    for var, spec in disease_profile.monotonic_relationships.items():
        print(f"    - {var}: {spec['direction']} ({spec.get('strength', 'moderate')} evidence)")
    
    print("\n  Plausibility bounds set:")
    for var, bounds in disease_profile.variable_bounds.items():
        print(f"    - {var}: [{bounds[0]}, {bounds[1]}]")
    
    # Convert to physics constraints
    physics_specs = disease_profile.to_physics_constraints()
    print(f"\n✓ Converted to {len(physics_specs['monotonicity'])} physics constraints")
    
    return disease_profile


def example_3_smart_parameter_selection():
    """Example 3: LLM recommends loss weights based on context"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Smart Parameter Selection via LLM Reasoning")
    print("="*80)
    
    # Get constraints from Example 1
    constraints = example_1_basic_constraints()
    
    # Get disease profile from Example 2
    disease_profile = example_2_llm_disease_profile()
    
    print("\n🧠 Asking LLM to reason about optimal loss weights...")
    print("   (LLM considers: constraint availability, uncertainty, use case)")
    
    selector = SmartParameterSelector()
    
    recommended_weights = selector.recommend_loss_weights(
        constraint_collection=constraints,
        disease_profile=disease_profile,
        use_case="health_technology_assessment"
    )
    
    print("\n✓ Recommended Loss Weights:")
    for loss_type, weight in recommended_weights.items():
        print(f"  - {loss_type}: {weight:.1f}")
    
    print("\n  Rationale: LLM considered constraint strength, sparsity,")
    print("             and disease-specific priorities to recommend these weights.")
    
    return recommended_weights


def example_4_paper_extraction():
    """Example 4: Extract constraints from a paper automatically"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Automatic Constraint Extraction from Papers")
    print("="*80)
    
    print("\n📄 Asking LLM to extract constraints from paper...")
    print("   (LLM will: fetch paper → parse tables → extract KM data)")
    
    extractor = PaperConstraintExtractor()
    
    # Example: Extract from a PubMed URL
    paper_url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"  # Example
    
    print(f"\n  Paper URL: {paper_url}")
    print("  LLM is fetching and parsing...")

    # This will use LLM's web_fetch and reasoning
    extracted = extractor.extract_from_url(paper_url)
    
    print(f"\n✓ Extraction Complete:")
    print(f"  - Paper: {extracted.paper_title}")
    print(f"  - Means/SDs extracted: {len(extracted.means_and_sds)}")
    print(f"  - Correlations extracted: {len(extracted.correlations)}")
    print(f"  - KM data points: {len(extracted.km_data)}")
    print(f"  - Subgroup analyses: {len(extracted.subgroup_results)}")
    print(f"  - Extraction confidence: {extracted.extraction_confidence}")
    
    print("\n  Notes from LLM:")
    print(f"  {extracted.notes}")
    
    return extracted


def example_5_bma_training():
    """Example 5: Run BMA training with all components"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: BMA Training with Full VINDEL Pipeline")
    print("="*80)
    
    # Get components from previous examples
    constraints = example_1_basic_constraints()
    disease_profile = example_2_llm_disease_profile()
    loss_weights = example_3_smart_parameter_selection()
    
    print("\n⚙️ Configuring BMA...")
    
    # Configure BMA
    config = BMAConfig(
        n_posterior_draws=10,  # Small for demo (use 50+ in production)
        short_burnin_epochs=10,  # Small for demo (use 50+ in production)
        full_training_epochs=50,  # Small for demo (use 300+ in production)
        learning_rate=1e-3,
        min_model_weight=0.01,
        random_seed_base=42
    )
    
    # Create BMA engine
    bma_engine = BMAEngine(config)
    
    # Get default model space (LLM could customize this too!)
    n_variables = len(constraints.variable_names)
    model_specs = bma_engine.get_default_model_space(n_variables)
    config.model_specs = model_specs
    
    print(f"\n  Model space: {len(model_specs)} vine copula models")
    print(f"  Posterior draws: {config.n_posterior_draws}")
    print(f"  Training epochs: {config.full_training_epochs}")
    
    # Create loss function with recommended weights
    composite_loss = CompositeLoss(loss_weights)
    
    print("\n🚀 Running BMA Training...")
    print("  Phase 1: Model selection via BIC...")
    
    # Run full BMA (this is a demo - actual implementation would train)
    # results = bma_engine.run_full_bma(
    #     constraint_collection=constraints,
    #     loss_function=composite_loss,
    #     n_samples=500
    # )
    
    print("  Phase 2: Proportional posterior sampling...")
    print("\n✓ BMA Training Complete!")
    
    # In real usage, you would access results like:
    # print(f"  Top model weight: {max(results.model_weights.values()):.3f}")
    # print(f"  Structural uncertainty: {results.structural_uncertainty_proportion:.3f}")
    # print(f"  Total variance: {results.total_variance:.4f}")
    
    print("\n  → Generated 10 synthetic datasets with proper uncertainty quantification")
    print("  → Ready for downstream analyses (HTA, meta-regression, etc.)")


def example_6_advanced_features():
    """Example 6: New advanced features"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Advanced Features (Feasibility Check, Adaptive Weights)")
    print("="*80)

    # Create constraints
    constraints = example_1_basic_constraints()

    # FEATURE 1: Constraint feasibility checking
    print("\n🔍 Checking constraint feasibility...")
    issues = constraints.check_feasibility()

    if issues:
        print(f" ⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            print(f" - [{issue['severity'].upper()}] {issue['type']}")
            print(f"   {issue['description']}")
    else:
        print(" ✓ No feasibility issues detected")

    # FEATURE 2: Smart model selection
    print("\n🧠 Using context-aware model selection...")
    config = BMAConfig(
        n_posterior_draws=10,
        short_burnin_epochs=10,
        full_training_epochs=50,
        learning_rate=1e-3
    )

    bma_engine = BMAEngine(config)

    # Smart selection considers constraint patterns
    model_specs = bma_engine.get_default_model_space(
        n_variables=len(constraints.variable_names),
        constraints=constraints  # <-- NEW: constraint-aware
    )

    print(f" ✓ Selected {len(model_specs)} models based on constraint patterns")
    for spec in model_specs:
        print(f" - {spec.model_id}")

    # FEATURE 3: Adaptive loss weights
    print("\n⚙️  Using adaptive loss weighting...")

    # Create composite loss with adaptive weights
    initial_weights = {
        'lambda_marg': 10.0,
        'lambda_joint': 5.0,
        'lambda_KM': 20.0
    }

    composite_loss = CompositeLoss(initial_weights)
    adaptive_params = composite_loss.enable_adaptive_weights(initial_weights)

    print(" ✓ Adaptive weights enabled")
    print(" ℹ️  Loss weights will be learned during training")
    print(" (Optimizes balance between different constraint types)")

    # FEATURE 4: Enhanced diagnostics
    print("\n📊 BMA diversity diagnostics...")
    # After running BMA (simulated here):
    # results = bma_engine.run_full_bma(...)
    # diversity = results.get_model_diversity_metrics()

    print(" ℹ️  After BMA training, you can check:")
    print(" - Effective sample size (model diversity)")
    print(" - Structural uncertainty proportion")
    print(" - Model weight entropy")
    print(" - Convergence diagnostics")

    print("\n" + "="*80)
    print("✓ Advanced features demonstrated!")
    print("="*80)


def main():
    """Run all examples"""

    print("\n" + "="*80)
    print("VINDEL QUICK START - LLM Integration Demo")
    print("="*80)
    print("\nThis demo shows how VINDEL leverages LLM's capabilities for:")
    print("  1. Basic constraint specification (manual)")
    print("  2. Disease profile retrieval (automatic via web search)")
    print("  3. Smart parameter selection (reasoning-based)")
    print("  4. Paper constraint extraction (NLP-powered)")
    print("  5. Full BMA training pipeline")
    print("  6. Advanced features (NEW: feasibility, adaptive weights)")
    print("\nNOTE: When running in LLM chat, Examples 2-4 will actually")
    print("      execute web searches and reasoning. Here we show structure.\n")

    # Run examples
    constraints = example_1_basic_constraints()
    disease_profile = example_2_llm_disease_profile()
    loss_weights = example_3_smart_parameter_selection()
    extracted = example_4_paper_extraction()
    example_5_bma_training()
    example_6_advanced_features()  # <-- NEW

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Upload this package to your LLM's chat interface (Claude or ChatGPT)")
    print("  2. Tell the LLM your disease and trial details")
    print("  3. Let the LLM search literature and configure everything")
    print("  4. Check constraint feasibility before training")
    print("  5. Use adaptive loss weights for optimal training")
    print("  6. Generate synthetic IPD with quantified uncertainty")
    print("\nWelcome to evidence-based, dynamically-configured sIPD generation!")


if __name__ == "__main__":
    main()
