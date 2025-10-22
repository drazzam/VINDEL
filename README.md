# VINDEL: VINe-based DEgree-of-freedom Learning

**LLM-Integrated Framework for Synthetic IPD Generation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

VINDEL is a **dynamic, evidence-based framework** for generating synthetic individual patient data (sIPD) from published aggregate trial summaries. Unlike static implementations, VINDEL is designed to work **within LLM chat interfaces (Claude or ChatGPT)**, leveraging the LLM's capabilities for:

- 🔍 **Real-time literature retrieval** via web search
- 🧠 **Context-aware parameter selection** using reasoning
- 📊 **Automatic constraint extraction** from papers
- 🏥 **Disease-specific customization** grounded in current evidence

**Important**: When uploading this package to your LLM (Claude or ChatGPT), please request the LLM to **load and activate Python runtime** to properly support loading these package files into the framework runtime.

## 🚀 Key Innovations

### 1. **Bayesian Model Averaging (BMA)**
- Properly quantifies structural uncertainty over vine copula models
- Two-phase algorithm: model selection → proportional posterior sampling
- Variance decomposition: `T_BMA = W̄ + (1+1/M)B_within + B_between`

### 2. **Constraint Enrichment Protocol**
- **Conditional correlations**: Subgroup-specific relationships
- **Subgroup survival**: Stratified Kaplan-Meier curves
- **Time-varying effects**: Non-proportional hazards
- **Multi-outcome consistency**: Binary + survival endpoint coherence

### 3. **Physics-Informed Constraints**
- **Monotonicity**: Age↑ → mortality↑ (if supported by literature, this is just an example not a rule)
- **Biological relationships**: Mediation, compositional constraints
- **Plausibility bounds**: Clinical guidelines-based ranges
- **Disease natural history**: Hazard profiles by disease type

### 4. **LLM Integration** (UNIQUE TO THIS PACKAGE)
- **Dynamic disease profiles**: Literature-retrieved, not hardcoded
- **Smart parameter selection**: Context-aware loss weights
- **Automatic constraint extraction**: Parse papers → constraints
- **Evidence transparency**: All recommendations cite sources

### 5. **Improved Loss Functions & Optimization** (NEW)
- **Fisher Z-transform**: Better gradient properties for correlation constraints
- **Quantile loss**: Proper asymmetric loss for quantile constraints
- **Adaptive loss weighting**: Learned balance between constraint types
- **Convergence diagnostics**: Automated training monitoring

### 6. **Constraint Feasibility Checking** (NEW)
- **Automatic validation**: Detects contradictory constraints before training
- **Positive definiteness**: Validates correlation matrices
- **Bound checking**: Ensures marginals respect plausibility bounds
- **Early warning system**: Prevents optimization failures

## 📦 Installation

```bash
# Clone or download this package
# Then install dependencies
pip install numpy scipy pandas torch scikit-learn matplotlib seaborn lifelines pyvinecopulib pydantic
```

**No pip installation needed** - upload this entire folder to your LLM's chat interface (Claude or ChatGPT)!

## 🎓 Usage Within LLM (Claude or ChatGPT)

### Step 1: Upload Package to Your LLM

1. Zip the `vindel_package` folder
2. Upload to your LLM chat (Claude or ChatGPT)
3. Request the LLM to load and activate Python runtime
4. The LLM will automatically detect the package structure

### Step 2: Configure Your Study

```python
from vindel import ConstraintCollection, MarginalConstraint, SurvivalConstraint

# Define your study constraints
constraints = ConstraintCollection(
    study_name="Example RCT",
    variable_names=["age", "weight", "bmi", "biomarker"],
    treatment_arms=["control", "treatment"]
)

# Add marginal constraints
constraints.marginal_constraints.append(
    MarginalConstraint(
        variable_name="age",
        mean=65.2,
        std=8.3,
        source="Table 1 of trial paper"
    )
)

# Add survival constraints (from Kaplan-Meier curves)
constraints.survival_constraints.append(
    SurvivalConstraint(
        arm="treatment",
        km_times=[0, 6, 12, 18, 24],  # months
        km_survival=[1.0, 0.85, 0.70, 0.55, 0.42],
        median_survival=18.5,
        source="Figure 2A of trial paper"
    )
)
```

### Step 3: Let Your LLM Enrich Constraints

**This is where the magic happens! Your LLM (Claude or ChatGPT) will:**

```python
from vindel.integration.llm_retriever import DiseaseProfileRetriever, PaperConstraintExtractor

# Tell your LLM about your disease
retriever = DiseaseProfileRetriever()
disease_profile = retriever.get_profile(
    disease_name="metastatic non-small cell lung cancer",
    additional_context={"line_of_therapy": "second-line", "histology": "non-squamous"}
)

# Your LLM will:
# 1. Search PubMed for "NSCLC natural history survival pattern"
# 2. Find median OS ~12 months, decreasing hazard (acute spike)
# 3. Identify prognostic biomarkers: PD-L1, EGFR, ALK
# 4. Set plausibility bounds from clinical guidelines
# 5. Return evidence-grounded DiseaseProfile with literature URLs

print(f"Disease profile confidence: {disease_profile.llm_confidence}")
print(f"Literature sources: {disease_profile.literature_sources}")
print(f"Hazard profile: {disease_profile.hazard_profile}")

# Apply physics constraints from disease profile
physics_constraints = disease_profile.to_physics_constraints()
# → Automatically creates:
#   - Monotonicity: age↑ → mortality↑ (if supported by literature)
#   - Bounds: ECOG [0-4], age [18-100], etc.
#   - Natural history: decreasing hazard for NSCLC
```

### Step 4: Extract Constraints from Papers

```python
# If you have a paper URL, let your LLM extract constraints automatically
extractor = PaperConstraintExtractor()

extracted = extractor.extract_from_url(
    "https://pubmed.ncbi.nlm.nih.gov/12345678/"
)

# Your LLM will:
# 1. Fetch full paper using web_fetch
# 2. Parse Tables 1-3 for baseline characteristics
# 3. Extract Kaplan-Meier data from Figure 2
# 4. Identify subgroup analyses from forest plots
# 5. Return structured ExtractedConstraints object

# Merge into your constraint collection
for marginal in extracted.means_and_sds:
    constraints.marginal_constraints.append(MarginalConstraint(...))
```

### Step 5: Run BMA Training

```python
from vindel.bma import BMAEngine, BMAConfig, VineStructure, CopulaFamily

# NEW: Check constraint feasibility before training
print("Checking constraint feasibility...")
issues = constraints.check_feasibility()

if issues:
    print("⚠️ WARNING: Detected potential issues:")
    for issue in issues:
        print(f" [{issue['severity']}] {issue['type']}: {issue['description']}")

    # Decide whether to proceed
    if any(issue['severity'] == 'error' for issue in issues):
        raise ValueError("Critical feasibility issues detected. Fix constraints before proceeding.")
else:
    print("✓ All constraints are feasible")

# Configure BMA with smart parameter selection
from vindel.integration.llm_retriever import SmartParameterSelector

param_selector = SmartParameterSelector()
loss_weights = param_selector.recommend_loss_weights(
    constraints, disease_profile, use_case="health_technology_assessment"
)

# Your LLM reasons:
# "Given 25 marginal constraints (strong), 10 correlations (moderate),
#  and 2 subgroup survival curves (high value), recommend:
#  λ_marg=12, λ_joint=6, λ_KM=25, λ_KM_subg=20, λ_physics=2"

# Create BMA engine
bma_engine = BMAEngine(BMAConfig(
    n_posterior_draws=50,
    short_burnin_epochs=50,
    full_training_epochs=300,
    learning_rate=1e-3
))

# NEW: Use smart model selection (context-aware)
model_specs = bma_engine.get_default_model_space(
    n_variables=len(constraints.variable_names),
    constraints=constraints  # Context-aware selection
)

# Configure BMA with enhanced features
config = BMAConfig(
    model_specs=model_specs,
    n_posterior_draws=50,
    short_burnin_epochs=50,
    full_training_epochs=300,
    learning_rate=1e-3
)

# Create engine and run
bma_engine = BMAEngine(config)
results = bma_engine.run_full_bma(
    constraint_collection=constraints,
    loss_function=composite_loss,
    n_samples=500
)

# BMA Results
print(f"Model weights: {results.summary()['model_weights']}")
print(f"Structural uncertainty: {results.structural_uncertainty_proportion:.3f}")

# Variance decomposition
print(f"Within-draw variance: {results.within_draw_variance:.4f}")
print(f"Within-model variance: {results.within_model_variance:.4f}")
print(f"Between-model variance: {results.between_model_variance:.4f}")
print(f"Total variance: {results.total_variance:.4f}")

# Interpretation
if results.structural_uncertainty_proportion > 0.30:
    print("⚠️ HIGH structural uncertainty - estimand is model-dependent")
elif results.structural_uncertainty_proportion > 0.10:
    print("✓ Moderate structural uncertainty - BMA essential")
else:
    print("✓ Low structural uncertainty - models agree")
```

### Step 6: Generate Synthetic IPD

```python
# After BMA training, generate multiple synthetic datasets
# (Implementation would generate from each posterior draw)

synthetic_datasets = []
for model_id, draws in results.posterior_samples.items():
    for draw in draws:
        # Generate synthetic data from this draw
        synthetic_data = generate_sipd_from_posterior(draw, constraints, n_samples=500)
        synthetic_datasets.append(synthetic_data)

# Now you have M=50 synthetic datasets, each honoring constraints
# Use for downstream analyses with proper uncertainty quantification
```

## 📊 What Makes This Package Different?

| Feature | Traditional sIPD Methods | VINDEL with LLM |
|---------|-------------------------|-------------------|
| **Disease profiles** | Hardcoded tables | Literature-retrieved |
| **Physics constraints** | Fixed rules | Evidence-based, disease-specific |
| **Constraint extraction** | Manual | Automated via LLM NLP |
| **Parameter selection** | Default values | Context-aware reasoning |
| **Evidence transparency** | None | Every decision cites sources |
| **Structural uncertainty** | Ignored | Quantified via BMA |
| **Updates** | Require code changes | Automatic via web search |

## ⚠️ Critical Limitations & Honest Reporting

### What VINDEL Provides
✅ Multiple plausible sIPD datasets reflecting epistemic + structural uncertainty  
✅ Exact reproduction of published aggregates (within tolerance)  
✅ Valid uncertainty quantification for **identified** estimands  
✅ Transparent handling of non-identified quantities  
✅ Evidence-based biological plausibility  

### What VINDEL Does NOT Provide
❌ Unique reconstruction of "true" patient-level data (mathematically impossible)  
❌ Discovery of associations not in published summaries  
❌ Replacement for real IPD in regulatory submissions (validation required)  
❌ Automatic detection of publication bias or data fabrication  

### Interpretation Guidelines

**For point-identified estimands** (e.g., published marginal distributions, overall survival):
- Report pooled estimates with confidence intervals
- Uncertainty reflects sampling + structural uncertainty
- Valid for inference (subject to validation)

**For non-identified estimands** (e.g., unpublished subgroup effects, tail dependence):
- Report **ranges** across posterior draws and sensitivity analyses
- Flag as **model-dependent**: "This estimate reflects modeling assumptions"
- Do NOT claim to "discover" patterns not in published data
- If `π_struct > 0.30`, be extra cautious

## 🤝 Appropriate Use Cases

✅ **Suitable for:**
- Health technology assessment when real IPD unavailable
- Methods development and benchmarking
- Exploratory heterogeneity analysis (hypothesis-generating)
- Network meta-regression with covariate adjustment
- Sample size calculations for future trials

❌ **Not suitable for:**
- Primary evidence for regulatory approval (without validation against real IPD)
- Claims of "discovering" subgroup effects not in published analyses
- Replacement for proper IPD meta-analysis when IPD accessible

## 🔧 Advanced Features

### Sensitivity Analysis

```python
# Test sensitivity to vine structure
sensitivity_configs = [
    {"model_specs": [cvine_only]},
    {"model_specs": [dvine_only]},
    {"model_specs": [mixed_vines]},
]

for config in sensitivity_configs:
    results = bma_engine.run_full_bma(constraints, loss_fn, n_samples)
    # Compare estimates across configurations
```

### Custom Physics Constraints

```python
from vindel.core.constraints import MonotonicityConstraint, PlausibilityBound

# Add domain-specific monotonicity
custom_monotone = MonotonicityConstraint(
    variable_name="creatinine_clearance",
    direction="decreasing",  # Higher CrCl → lower mortality risk
    outcome="mortality",
    evidence_source="KDIGO guidelines 2024",
    evidence_strength="strong"
)

physics_constraints.monotonicity_constraints.append(custom_monotone)
```

## 🐛 Troubleshooting

**Q: The LLM isn't executing web searches**
A: Make sure you've uploaded the package to your LLM's chat interface (Claude or ChatGPT) and requested it to load Python runtime. The LLM detects the `ClaudeToolInterface` calls and intercepts them.

**Q: BMA shows high structural uncertainty (π_struct > 0.5)**  
A: This means your estimand is highly model-dependent. Consider:
1. Adding more enriched constraints to reduce degrees of freedom
2. Reporting results as ranges across models
3. Flagging estimand as non-identified in your analysis

**Q: Physics constraints being violated**  
A: Increase physics constraint weights (`λ_physics`) or check for data-constraint conflicts (infeasibility).

## 📄 License

MIT License - see LICENSE file

## 👨‍💻 Author

Ahmed Y. Azzam, MD, MEng, DSc(h.c.), FRCP

## 🙏 Acknowledgments

Built on foundations from:
- Vine copula theory (Aas et al., 2009; Dissmann et al., 2013)
- IPDfromKM methods (Guyot et al., 2012; Wei & Royston, 2017)
- Bayesian model averaging (Hoeting et al., 1999)
- Physics-informed learning paradigms

**Unique contribution**: First sIPD framework to integrate real-time literature retrieval and evidence-based customization through LLM capabilities.

---

## 🚀 Quick Start (30 seconds)

```python
# 1. Upload this package to your LLM (Claude or ChatGPT)
# 2. Request the LLM to load and activate Python runtime
# 3. Tell the LLM your disease and constraints
# 4. Let the LLM search literature and configure everything
# 5. Run BMA training
# 6. Generate M=50 synthetic datasets with proper uncertainty

# That's it! Your LLM handles all the complexity.
```

**Welcome to evidence-based, dynamically-configured synthetic data generation!**

---

## 📚 COMPREHENSIVE FRAMEWORK DOCUMENTATION

### Overview of VINDEL Architecture

VINDEL is a sophisticated framework that combines multiple advanced techniques to generate high-quality synthetic individual patient data from aggregate statistics. This section provides a detailed explanation of the framework's architecture and components.

### VINDEL Framework Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        VINDEL FRAMEWORK ARCHITECTURE                              │
│                VINe-based DEgree-of-freedom Learning System                       │
└─────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 1: INPUT & CONSTRAINT COLLECTION                                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐
│  Manual Input    │  │  LLM Retrieval   │  │ Paper Extraction │  │ Literature  │
│  • Table 1       │  │  • PubMed search │  │ • Tables parsing │  │ • Guidelines│
│  • KM curves     │  │  • Disease prof. │  │ • KM digitization│  │ • Domain exp│
│  • HRs, CIs      │  │  • Biomarkers    │  │ • Subgroup data  │  │ • Priors    │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └──────┬──────┘
         │                     │                     │                    │
         └─────────────────────┴─────────────────────┴────────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │  CONSTRAINT COLLECTION         │
                        │  ┌──────────────────────────┐  │
                        │  │ Marginal (C_marg)        │  │  11 Constraint Types:
                        │  │ Joint (C_joint)          │  │  ─────────────────────
                        │  │ Conditional (C_cond)     │  │  • Marginal distributions
                        │  │ Survival (C_surv)        │  │  • Pairwise correlations
                        │  │ Subgroup Surv.           │  │  • Conditional correlations
                        │  │ Time-varying HR          │  │  • Survival curves (KM)
                        │  │ Multi-outcome            │  │  • Subgroup survival
                        │  │ Causal (C_causal)        │  │  • Time-varying hazards
                        │  │ Network (C_net)          │  │  • Multi-outcome coherence
                        │  │ Physics (C_phys)            │  │  • Treatment effects
                        │  └──────────────────────────┘  │  • Network consistency
                        └────────────────┬───────────────┘  • Physics-informed
                                         │
                                         ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 2: VALIDATION & FEASIBILITY CHECKING                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

              ┌───────────────────────────────────────────────┐
              │  ConstraintFeasibilityChecker                 │
              │  ┌─────────────────────────────────────────┐  │
              │  │ ✓ Marginal-Bound Conflicts              │  │
              │  │ ✓ Correlation Matrix Validity (PD)      │  │
              │  │ ✓ Survival-Marginal Consistency         │  │
              │  │ ✓ Subgroup-Marginal Coherence           │  │
              │  └─────────────────────────────────────────┘  │
              │  Issues? → {type, severity, description}      │
              └──────────────────┬────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  All Feasible?          │
                    ├─────────────────────────┤
                    │ YES → Proceed           │
                    │ WARNINGS → User decides │
                    │ ERRORS → Stop & fix     │
                    └────────────┬────────────┘
                                 │
                                 ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 3: LOSS FUNCTION CONSTRUCTION                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPOSITE LOSS: L_total = Σᵢ λᵢ · Lᵢ                                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ L₁: Marginal Loss           → λ_marg · Σ (μ_emp - μ_target)²         │  │
│  │ L₂: Joint Loss (Fisher Z)      → λ_joint · Σ (z_emp - z_target)²     │  │
│  │ L₃: Conditional Loss        → λ_cond · Σ (ρ|subgroup - ρ_target)²    │  │
│  │ L₄: KM Loss                 → λ_KM · Σ (S(t)_emp - S(t)_pub)²        │  │
│  │ L₅: Subgroup KM Loss         → λ_KM,subg · Σ (S_g(t) - S_g,pub(t))²  │  │
│  │ L₆: Time-varying HR          → λ_HR,TV · Σ (log HR_k - log HR_k^pub)²│  │
│  │ L₇: Quantile Loss            → λ_quantile · Σ check(τ, q_emp, q_tgt) │  │
│  │ L₈: Multi-outcome           → λ_multi · (binary + survival + mono)   │  │
│  │ L₉: Causal Loss             → λ_causal · (ATE_emp - ATE_pub)²        │  │
│  │ L₁₀: Physics Loss           → λ_phys · (monotone + bounds + ...  )   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ ADAPTIVE WEIGHTING                                                    │  │
│  │ • Learn λ via softmax parameterization                                │  │
│  │ • Entropy regularization prevents collapse                            │  │
│  │ • Automatic balance based on constraint strength                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 4: MODEL SPACE SELECTION                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

              ┌────────────────────────────────────────────┐
              │ CONTEXT-AWARE MODEL SELECTION              │
              │                                             │
              │ Analyze Constraints:                       │
              │ ├─ Has heavy survival? → Add Archimedean  │
              │ ├─ High correlations? → Add Student-t      │
              │ ├─ Conditional deps? → Add mixed families  │
              │ └─ High dimension? → Reduce model space    │
              │                                             │
              │ Generate Model Space M = {m₁, ..., mₖ}:   │
              │ ┌─────────────────────────────────────┐    │
              │ │ m₁: C-Vine + Gaussian               │    │
              │ │ m₂: C-Vine + Student-t              │    │
              │ │ m₃: C-Vine + Clayton (if survival)  │    │
              │ │ m₄: C-Vine + Gumbel (if survival)   │    │
              │ │ m₅: C-Vine + Mixed                  │    │
              │ │ m₆: D-Vine + Mixed (if p≤15)        │    │
              │ │ m₇: R-Vine + Gaussian (if p≤10)     │    │
              │ │ m₈: Independence (null model)       │    │
              │ └─────────────────────────────────────┘    │
              └──────────────────┬─────────────────────────┘
                                 │
                                 ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 5: BAYESIAN MODEL AVERAGING (BMA) - TWO-PHASE ALGORITHM                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Model Selection (Short Burn-In)                                     │
│                                                                               │
│  For each model m ∈ M:                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ 1. Initialize θ_m ~ p(θ|m)                                         │     │
│  │ 2. Train for E_short epochs (e.g., 50)                             │     │
│  │ 3. Compute BIC_m = 2·L(θ_m) + k_m·log(n)                           │     │
│  │    where k_m = # parameters in model m                             │     │
│  │ 4. Calculate weight: w_m ∝ exp(-BIC_m/2)                           │     │
│  │ 5. Normalize: w_m ← w_m / Σⱼ w_j                                   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
│  Result: Model posterior weights {w₁, ..., wₖ}                              │
│  ┌────────────┬─────────┬─────────┐                                         │
│  │ Model      │ BIC     │ Weight  │                                         │
│  ├────────────┼─────────┼─────────┤                                         │
│  │ C+Gaussian │ 1250.3  │ 0.42    │  ← Dominant model                       │
│  │ C+Student-t│ 1252.1  │ 0.31    │                                         │
│  │ C+Mixed    │ 1255.8  │ 0.18    │                                         │
│  │ D+Mixed    │ 1259.2  │ 0.07    │                                         │
│  │ Independence│ 1275.9  │ 0.02    │  ← Pruned (w < 0.01)                   │
│  └────────────┴─────────┴─────────┘                                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Proportional Posterior Sampling                                     │
│                                                                               │
│  Allocate draws: M_m = ⌈M · w_m⌉  (e.g., M=50 total draws)                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ m=C+Gaussian (w=0.42): M₁ = 21 draws                            │        │
│  │ ├─ Draw 1: θ₁⁽¹⁾ ~ p(θ|m₁, C) via full training (E_full=300)    │        │
│  │ ├─ Draw 2: θ₁⁽²⁾ ~ p(θ|m₁, C) [independent initialization]       │        │
│  │ └─ ... (21 draws total)                                          │        │
│  │                                                                   │        │
│  │ m=C+Student-t (w=0.31): M₂ = 16 draws                            │        │
│  │ m=C+Mixed (w=0.18): M₃ = 9 draws                                 │        │
│  │ m=D+Mixed (w=0.07): M₄ = 4 draws                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                               │
│  Result: {θ⁽¹⁾, ..., θ⁽ᴹ⁾} ~ BMA posterior                                  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 6: SYNTHETIC IPD GENERATION & UNCERTAINTY QUANTIFICATION                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ Generate M Synthetic Datasets:                                               │
│                                                                               │
│  For each draw θ⁽ʲ⁾:                                                         │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │ 1. Sample from vine copula: U ~ C(u|θ⁽ʲ⁾)                  │             │
│  │ 2. Apply marginal transforms: X = F⁻¹(U)                   │             │
│  │ 3. Generate n=500 patient records → Dataset D⁽ʲ⁾           │             │
│  └────────────────────────────────────────────────────────────┘             │
│                                                                               │
│  Result: {D⁽¹⁾, ..., D⁽ᴹ⁾}  (M synthetic IPD datasets)                      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ VARIANCE DECOMPOSITION (Honest Uncertainty Quantification)                   │
│                                                                               │
│  For any estimand ψ (e.g., treatment effect):                               │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │ Compute ψ̂⁽ʲ⁾ and Var(ψ̂⁽ʲ⁾) for each dataset j=1,...,M    │             │
│  │                                                             │             │
│  │ Decompose total variance:                                  │             │
│  │ T_BMA = W̄ + (1 + 1/M)·B_within + B_between                │             │
│  │                                                             │             │
│  │ where:                                                      │             │
│  │ • W̄ = Σⱼ Var(ψ̂⁽ʲ⁾)/M         (within-draw variance)        │             │
│  │ • B_within = model-specific posterior variance             │             │
│  │ • B_between = Σ_m w_m(ψ̄_m - ψ̄_BMA)²  (structural uncert.) │             │
│  │                                                             │             │
│  │ Structural uncertainty proportion:                         │             │
│  │ π_struct = B_between / T_BMA                               │             │
│  └────────────────────────────────────────────────────────────┘             │
│                                                                               │
│  Interpretation:                                                             │
│  ├─ π < 0.10: Single model sufficient, low structural sensitivity           │
│  ├─ 0.10 ≤ π < 0.30: BMA recommended, moderate structural uncert.           │
│  └─ π ≥ 0.30: BMA ESSENTIAL, estimand is model-dependent                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ LAYER 7: VALIDATION & DIAGNOSTICS                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ SyntheticIPDValidator                                                         │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 1. Aggregate Reproduction                                              │   │
│ │    ├─ Marginals: max relative error < 5%                               │   │
│ │    ├─ Correlations: max absolute error < 0.05                          │   │
│ │    └─ Survival curves: pointwise error < 0.10                          │   │
│ │                                                                         │   │
│ │ 2. Distributional Properties                                           │   │
│ │    ├─ Normality tests (Shapiro-Wilk)                                   │   │
│ │    ├─ Outlier detection (±3 SD)                                        │   │
│ │    └─ Skewness & kurtosis checks                                       │   │
│ │                                                                         │   │
│ │ 3. BMA Diagnostics (NEW!)                                              │   │
│ │    ├─ Effective Sample Size: ESS = 1/Σw²                               │   │
│ │    ├─ Shannon entropy: H = -Σw·log(w)                                  │   │
│ │    ├─ Model diversity metrics                                          │   │
│ │    └─ Convergence diagnostics                                          │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│ Generate Validation Report: ✓ PASS / ✗ FAIL                                 │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║ OUTPUT: M VALIDATED SYNTHETIC IPD DATASETS                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ Ready for Downstream Analysis:                                               │
│ ├─ Health Technology Assessment (HTA)                                        │
│ ├─ Network Meta-Analysis (NMA) / Meta-Regression                            │
│ ├─ Subgroup Analysis (with proper uncertainty)                              │
│ ├─ Sample Size Calculations                                                 │
│ └─ Methods Development & Benchmarking                                        │
│                                                                               │
│ With Honest Uncertainty Quantification:                                      │
│ ├─ Point estimates: ψ̄_BMA = Σⱼ ψ̂⁽ʲ⁾/M                                      │
│ ├─ Total variance: T_BMA (sampling + structural)                            │
│ ├─ 95% CI: ψ̄_BMA ± 1.96·√T_BMA                                              │
│ └─ Structural uncertainty flag if π_struct > 0.30                            │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
KEY INNOVATIONS IN VINDEL:
───────────────────────────────────────────────────────────────────────────────
1. 11 constraint types (4 NEW: conditional, subgroup surv., time-varying HR, multi-outcome)
2. Physics-informed constraints (monotonicity, bounds, natural history)
3. Bayesian Model Averaging for structural uncertainty quantification
4. LLM integration for dynamic constraint enrichment from literature
5. Fisher Z-transform for improved correlation loss gradients 
6. Quantile loss with check function 
7. Adaptive loss weighting via learnable parameters 
8. Constraint feasibility checking before training 
9. Context-aware model selection based on constraint patterns 
10. Enhanced BMA diagnostics (ESS, entropy, diversity metrics) 
═══════════════════════════════════════════════════════════════════════════════
```

### Core Components

#### 1. **Constraint System**

The VINDEL framework supports 11 different constraint types, organized into four categories:

**A. Basic Constraints**
- **Marginal Constraints** (`C_marg`): Enforce univariate distributions (mean, std, quantiles)
  - Example: Age ~ N(65, 10), Weight median = 72kg
  - Loss: L₁ squared error between empirical and target statistics

- **Joint Constraints** (`C_joint`): Enforce bivariate dependencies
  - Pearson correlation, Kendall's tau, Spearman's rho
  - NEW: Fisher Z-transform for better gradient properties
  - Loss: Squared error in Fisher Z-space with uncertainty weighting

- **Survival Constraints** (`C_surv`): Match Kaplan-Meier curves
  - Time-specific survival probabilities
  - Median survival, hazard ratios
  - Loss: Sum of squared deviations at specified time points

**B. Enrichment Constraints **
- **Conditional Correlations** (`C_cond`): Subgroup-specific correlations
  - Example: Age-Weight correlation differs by gender
  - Enables capturing heterogeneous dependence structures

- **Subgroup Survival** (`C_surv,subg`): Stratified survival curves
  - Example: PD-L1+ patients have different survival than PD-L1-
  - Critical for precision medicine applications

- **Time-Varying Hazard Ratios** (`C_HR,TV`): Non-proportional hazards
  - Example: Immunotherapy effect increases over time
  - Enables realistic modeling of delayed treatment effects

- **Multi-Outcome Consistency** (`C_multi`): Cross-endpoint coherence
  - Example: Responders have better survival than non-responders
  - Enforces biological plausibility across endpoints

**C. Network Constraints** (`C_net`)
- Network meta-analysis consistency
- Indirect comparison constraints
- Heterogeneity modeling

**D. Physics-Informed Constraints** (`C_phys`)
- **Monotonicity**: Age ↑ → mortality ↑ (if biologically supported)
- **Plausibility Bounds**: Clinical guidelines-based ranges
- **Natural History**: Disease-specific hazard profiles
- **Biological Relationships**: Mediation, compositional constraints

#### 2. **Loss Function System**

The composite loss function combines all constraints:

```
L_total = Σᵢ λᵢ · Lᵢ

where:
- L_marg: Marginal distribution loss
- L_joint: Joint dependence loss (with Fisher Z-transform)
- L_cond: Conditional correlation loss
- L_KM: Kaplan-Meier fidelity loss
- L_KM,subg: Subgroup survival loss
- L_HR,TV: Time-varying HR loss
- L_multi: Multi-outcome consistency loss
- L_causal: Treatment effect loss
- L_phys: Physics-informed regularization
```

**NEW: Enhanced Loss Components**

1. **Fisher Z-Transform for Correlations**
   - Standard approach: minimize (ρ_emp - ρ_target)²
   - Problem: Correlation near ±1 have poor gradient properties
   - Solution: z = arctanh(ρ), which has approximately normal distribution
   - Benefit: Improved optimization stability and convergence

2. **Quantile Loss (Check Function)**
   - Standard approach: minimize (q_emp - q_target)²
   - Problem: Symmetric loss doesn't respect quantile definition
   - Solution: L(τ) = τ·max(0, y-ŷ) + (1-τ)·max(0, ŷ-y)
   - Benefit: Proper asymmetric loss for quantile constraints

3. **Adaptive Loss Weighting**
   - Standard approach: Fixed λ weights specified by user
   - Problem: Optimal weights depend on constraint strength and data
   - Solution: Learn λ via softmax parameterization during training
   - Benefit: Automatic balance between competing constraints

#### 3. **Bayesian Model Averaging (BMA) Engine**

The BMA engine is the core of VINDEL's uncertainty quantification:

**Phase 1: Model Selection (Short Burn-In)**
1. Train K vine copula models with different structures (C-vine, D-vine, R-vine)
2. Compute BIC for each model: BIC_m = 2·L(θ_m) + k_m·log(n)
3. Calculate model weights: w_m ∝ exp(-BIC_m/2)
4. Prune models with w_m < threshold (default: 0.01)

**Phase 2: Proportional Posterior Sampling**
1. Allocate draws proportionally: M_m = ⌈M · w_m⌉
2. For each model m, draw M_m independent posterior samples
3. Each sample undergoes full training (E_full epochs)
4. Result: M total posterior draws from model-averaged distribution

**Variance Decomposition**

VINDEL provides honest uncertainty quantification via:

```
T_BMA = W̄ + (1 + 1/M)·B_within + B_between

where:
- W̄: Average within-draw variance (sampling uncertainty)
- B_within: Between-draw, within-model variance (MCMC uncertainty)
- B_between: Between-model variance (structural uncertainty)
```

**Structural Uncertainty Proportion**: π_struct = B_between / T_BMA
- π < 0.10: Single model sufficient
- 0.10 ≤ π < 0.30: BMA recommended
- π ≥ 0.30: Estimand is structurally sensitive

**NEW: Enhanced BMA Diagnostics**

1. **Effective Sample Size (ESS)**
   - ESS = 1 / Σ w_m²
   - ESS = 1: Single model dominates
   - ESS = M: Uniform weights (maximum diversity)
   - ESS < 3: Warning - model space may be too narrow

2. **Model Diversity Metrics**
   - Shannon entropy: H = -Σ w_m·log(w_m)
   - Gini coefficient: Measures weight inequality
   - Normalized entropy: H / log(M) ∈ [0, 1]

3. **Convergence Diagnostics**
   - Moving average convergence
   - Trend analysis (should be decreasing)
   - Variance stability checks
   - Epochs since last improvement

**NEW: Context-Aware Model Selection**

The enhanced `get_default_model_space` method adapts the model space based on problem characteristics:

- **High correlations** (max |ρ| > 0.6): Add Student-t copulas for heavy tails
- **Heavy survival constraints** (>3 KM curves): Add Archimedean copulas (Clayton, Gumbel)
- **Conditional constraints**: Suggests tail dependence → mixed families
- **High dimensionality** (p > 20): Reduce model space for computational efficiency

#### 4. **Constraint Feasibility Checker**

Before training, VINDEL checks for contradictory constraints:

**1. Marginal-Bound Conflicts**
- Check if mean ± 3·SD exceeds plausibility bounds
- Example: Age mean=65, SD=20 but bounds=[18, 100]
- Verdict: Warning (3·SD reach would include negative ages)

**2. Correlation Matrix Validity**
- Build correlation matrix from pairwise constraints
- Check positive definiteness via eigenvalue analysis
- Example: ρ(A,B)=0.9, ρ(B,C)=0.9, ρ(A,C)=-0.9
- Verdict: Error (impossible correlation structure)

**3. Survival-Marginal Consistency**
- Check if median survival is consistent with KM curve
- Example: Median survival = 18 months, but S(18) = 0.7
- Verdict: Warning (median should have S(t) ≈ 0.5)

**4. Subgroup-Marginal Coherence**
- Verify subgroup ranges are compatible with overall distribution
- Example: Subgroup mean outside overall 95% CI
- Verdict: Warning (potential incompatibility)

#### 5. **Vine Copula Models**

Vine copulas decompose multivariate dependencies into bivariate building blocks:

**C-Vine (Canonical Vine)**
- Star structure: one root variable connected to all others
- Best for: Data with a dominant central variable
- Computational: O(p²) pairs

**D-Vine (Drawable Vine)**
- Path structure: sequential conditioning
- Best for: Natural ordering (e.g., temporal variables)
- Computational: O(p) pairs per level

**R-Vine (Regular Vine)**
- General structure: any valid vine specification
- Best for: Complex dependence structures
- Computational: Most expensive, O(p²) pairs

**Copula Families**
- **Gaussian**: Symmetric, tail independent
- **Student-t**: Symmetric, heavy tails, tail dependent
- **Clayton**: Lower tail dependent (joint lows)
- **Gumbel**: Upper tail dependent (joint highs)
- **Frank**: Symmetric, weak tail dependence

#### 6. **LLM Integration**

VINDEL uniquely leverages LLM capabilities:

**A. Disease Profile Retrieval**
- LLM searches medical literature (PubMed, clinical guidelines)
- Extracts: median survival, hazard profiles, prognostic factors
- Recommends: plausibility bounds, monotonicity constraints
- Cites: All sources for transparency

**B. Smart Parameter Selection**
- Analyzes constraint availability and uncertainty
- Reasons about use case (HTA vs. methods development)
- Recommends loss weights (λ_marg, λ_joint, λ_KM, etc.)
- Explains rationale in natural language

**C. Automatic Constraint Extraction**
- Fetches papers via URL or PMID
- Parses: Tables 1-3 (baseline characteristics)
- Digitizes: Kaplan-Meier curves from figures
- Extracts: Subgroup analyses, forest plots
- Returns: Structured constraints with confidence scores

### Theoretical Foundations

#### Identifiability and Honest Uncertainty

VINDEL explicitly acknowledges the identifiability hierarchy:

1. **Identified Estimands**
   - Published marginal distributions
   - Published pairwise correlations
   - Overall survival curves
   - Treatment effects (ATE)
   → BMA quantifies sampling + structural uncertainty

2. **Partially Identified Estimands**
   - Conditional correlations (if subgroup-specific data available)
   - Tail dependence (if relevant copula constraints specified)
   → BMA provides ranges, flags model-dependence

3. **Non-Identified Estimands**
   - Unpublished subgroup effects
   - Individual-level treatment effect heterogeneity
   - Complex causal structures
   → Report as model-dependent, DO NOT claim discovery

**Interpretation Guidelines**:
- π_struct < 0.10: Point estimates + CI (low structural sensitivity)
- π_struct ≥ 0.30: Report ranges across models, flag as "model-dependent"
- Always validate against real IPD when available

#### Optimization Principles

**Gradient-Based Training**
- Use automatic differentiation (PyTorch)
- Fisher Z-transform improves gradient flow for correlations
- Adaptive weights prevent gradient starvation
- Early stopping prevents overfitting to noise

**Convergence Diagnostics**
- Monitor: loss trajectory, moving average, variance
- Detect: plateaus, oscillations, divergence
- Adaptive learning rate scheduling
- Gradient clipping for stability

#### Validation and Diagnostics

**Post-Training Validation**
- Aggregate reproduction: max relative error < 5%
- Correlation accuracy: max absolute error < 0.05
- Survival curve fidelity: max pointwise error < 0.10
- Plausibility checks: no violations of bounds or monotonicity

**Distributional Diagnostics**
- Shapiro-Wilk test for normality
- Outlier detection (±3 SD)
- Skewness and kurtosis checks
- Multimodality detection

### Usage Patterns and Best Practices

#### When to Use VINDEL

✅ **Appropriate Use Cases**:
- Health technology assessment when IPD unavailable
- Network meta-regression with covariate adjustment
- Sample size calculations for future trials
- Methods development and benchmarking
- Exploratory heterogeneity analysis (hypothesis-generating)

❌ **Inappropriate Use Cases**:
- Primary evidence for regulatory approval (without validation) unless there are sufficient trials that prove first instance of safety and efficacy (under caution keep in mind)
- Claims of "discovering" effects not in published data
- Replacement for real IPD when accessible
- Individual patient-level predictions

#### Workflow Recommendations

1. **Constraint Collection**
   - Start with published aggregates (Table 1, KM curves)
   - Use LLM to enrich from literature
   - Add physics constraints from domain knowledge
   - Run feasibility checker before training

2. **Model Configuration**
   - Use context-aware model selection
   - Start with default loss weights, enable adaptive weighting
   - Allocate ≥50 posterior draws for BMA
   - Enable convergence diagnostics

3. **Training and Validation**
   - Monitor convergence via diagnostics
   - Check aggregate reproduction after training
   - Compute structural uncertainty proportion
   - Validate distributional properties

4. **Inference and Reporting**
   - Report BMA-pooled estimates with total variance
   - Flag estimands with π_struct > 0.30
   - Provide ranges for non-identified quantities
   - Conduct sensitivity analyses

#### Advanced Features

**Custom Physics Constraints**
```python
from vindel import MonotonicityConstraint, PlausibilityBound

# Add domain-specific monotonicity
custom_monotone = MonotonicityConstraint(
    variable_name="creatinine_clearance",
    direction="decreasing",  # Higher CrCl → lower mortality
    outcome="mortality",
    evidence_source="KDIGO guidelines 2024",
    evidence_strength="strong"
)
```

**Adaptive Loss Weighting**
```python
composite_loss = CompositeLoss(initial_weights)
adaptive_params = composite_loss.enable_adaptive_weights(initial_weights)

# Optimize both model parameters and loss weights
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(adaptive_params),
    lr=1e-3
)
```

**Enhanced Diagnostics**
```python
# After BMA training
ess = results.compute_effective_sample_size()
diversity = results.get_model_diversity_metrics()

print(f"ESS: {ess:.1f}")
print(f"Interpretation: {diversity['interpretation']}")
```

### Technical Specifications

**Computational Requirements**
- CPU-only implementation (no GPU required)
- Memory: ~2GB for p ≤ 20 variables, n = 500 samples
- Time: ~5-10 minutes for full BMA with M = 50 draws (on modern CPU)

**Dependencies**
- Python 3.10+
- PyTorch (CPU version)
- NumPy, SciPy, pandas
- scikit-learn (for preprocessing)
- pyvinecopulib (vine copula library)
- lifelines (survival analysis)
- pydantic (data validation)

**Limitations**
- Maximum recommended variables: p ≤ 30
- Computational cost: O(p²·M·E) for p variables, M draws, E epochs
- Memory: O(n·p + p²) for n samples, p variables
- Assumes MAR (missing at random) for missing data

### Troubleshooting

**Issue**: High structural uncertainty (π_struct > 0.5)
- **Cause**: Estimand is highly model-dependent
- **Solution**: Add more enrichment constraints, or report as range

**Issue**: BIC favors independence model
- **Cause**: Constraints are very sparse or conflicting
- **Solution**: Check feasibility, add joint constraints

**Issue**: Training diverges or oscillates
- **Cause**: Conflicting constraints, poor initialization, high learning rate
- **Solution**: Reduce LR, check feasibility, use gradient clipping

**Issue**: Aggregate reproduction poor (error > 10%)
- **Cause**: Infeasible constraints, insufficient training, wrong model space
- **Solution**: Run feasibility checker, increase epochs, expand model space

### References and Further Reading

**Vine Copulas**
- Aas et al. (2009): "Pair-copula constructions of multiple dependence"
- Dissmann et al. (2013): "Selecting and estimating regular vine copulae"

**Bayesian Model Averaging**
- Hoeting et al. (1999): "Bayesian model averaging: a tutorial"
- Raftery et al. (2005): "Using Bayesian model averaging to calibrate forecasts"

**Synthetic IPD Methods**
- Guyot et al. (2012): "Enhanced secondary analysis via IPD reconstruction"
- Wei & Royston (2017): "Reconstructing time-to-event data from published KM curves"

**VINDEL Framework**
- GitHub repository: https://github.com/drazzam/VINDEL

### Citation

If you use VINDEL in your research, please cite:

```bibtex
@software{vindel2025,
  author = {Azzam, Ahmed Y.},
  title = {VINDEL: VINe-based DEgree-of-freedom Learning for Synthetic IPD Generation},
  year = {2025},
  url = {https://github.com/drazzam/VINDEL}
}
```

### Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

### License

MIT License - see LICENSE file for details.

### Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Email: ahmed.azzam@hsc.wvu.edu

---

**VINDEL: Bringing Evidence-Based Rigor to Synthetic IPD Generation**
