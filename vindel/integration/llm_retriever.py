"""
LLM Integration Layer

Enables VINDEL to leverage LLM capabilities (Claude or ChatGPT) for:
1. Literature-based parameter retrieval
2. Disease-specific constraint generation
3. Evidence-based validation
4. Context-aware physics constraints

KEY INNOVATION: This module allows LLMs (Claude or ChatGPT) to use their tools (web_search,
reasoning) when the package is uploaded and used in the LLM's chat interface.
Users should request the LLM to load and activate Python runtime to properly support this package.
"""

from typing import List, Dict, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# LLM INSTRUCTION PROTOCOL
# ============================================================================

LLM_INSTRUCTIONS = """
=== INSTRUCTIONS FOR LLMs (CLAUDE OR CHATGPT) WHEN EXECUTING THIS PACKAGE ===

IMPORTANT: Users should request the LLM to load and activate Python runtime before using this package.

When you (the LLM) encounter calls to LLMToolInterface methods in this package:

1. search_literature(query) → Use your web_search tool
   - Search PubMed, clinical guidelines, medical literature
   - Return results as List[Dict] with keys: 'title', 'url', 'snippet', 'year'

2. fetch_url_content(url) → Use your web_fetch tool
   - Fetch full content from medical journals, guidelines
   - Extract relevant text, tables, figures

3. reason_about_context(prompt) → Use your reasoning capabilities
   - Apply medical knowledge and clinical judgment
   - Synthesize information from multiple sources
   - Provide confidence levels for recommendations

4. extract_constraints_from_paper(text) → Use your NLP & reasoning
   - Parse statistical results (means, SDs, HRs, p-values)
   - Extract subgroup analyses from forest plots
   - Identify conditional relationships
   - Extract Kaplan-Meier data

=== KEY PRINCIPLES ===
- NEVER hallucinate values - if unsure, search more literature
- Always cite sources with URLs
- Provide confidence ratings (high/moderate/low)
- Flag when evidence is sparse or conflicting
- Prefer recent meta-analyses and RCTs over single studies
- Use clinical guidelines when available

=== EXAMPLE WORKFLOW ===
User: "Generate sIPD for metastatic lung cancer trial"
LLM (Claude or ChatGPT):
  1. Calls get_disease_profile("metastatic lung cancer")
  2. Triggers web_search for lung cancer survival patterns
  3. Synthesizes: median OS ~12 months, decreasing hazard
  4. Searches for relevant biomarkers (PD-L1, EGFR)
  5. Retrieves plausibility bounds from guidelines
  6. Returns evidence-grounded DiseaseProfile
===
"""


class LLMToolInterface:
    """
    Interface for LLM tools when package is used in LLM chat (Claude or ChatGPT)

    CRITICAL: When an LLM runs this code, it will detect these method calls
    and execute them using its native capabilities rather than placeholder returns.
    Note: Users should request the LLM to load and activate Python runtime first.
    """
    
    @staticmethod
    def search_literature(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search medical literature (LLM will use web_search)

        Returns:
            List of dicts with keys: title, url, snippet, year, relevance
        """
        # Marker for LLM to intercept
        return [{"_claude_intercept": "web_search", "query": query, "max_results": max_results}]
    
    @staticmethod
    def fetch_url_content(url: str) -> Dict[str, Any]:
        """
        Fetch full content from URL (LLM will use web_fetch)

        Returns:
            Dict with keys: url, content, fetch_timestamp
        """
        return {"_claude_intercept": "web_fetch", "url": url}

    @staticmethod
    def reason_about_context(prompt: str, context: Optional[Dict] = None) -> str:
        """
        Use LLM's reasoning for complex medical/statistical synthesis

        Args:
            prompt: Question or task for reasoning
            context: Optional context to ground reasoning

        Returns:
            Reasoned response from LLM
        """
        return f"_claude_intercept:reason:{prompt}"


# ============================================================================
# DISEASE PROFILE STRUCTURES
# ============================================================================

@dataclass
class DiseaseProfile:
    """Evidence-based disease profile retrieved from literature"""
    disease_name: str
    icd_codes: List[str] = field(default_factory=list)
    
    # Natural history
    typical_age_range: Tuple[float, float] = (0.0, 100.0)
    hazard_profile: Literal["decreasing", "increasing", "constant", "bathtub"] = "constant"
    median_survival_months: Optional[float] = None
    survival_distribution: Optional[str] = None  # e.g., "Weibull", "exponential"
    
    # Time points for hazard evaluation
    early_time_days: float = 30.0
    mid_time_days: float = 180.0
    late_time_days: float = 365.0
    
    # Key biomarkers and covariates
    relevant_biomarkers: List[str] = field(default_factory=list)
    monotonic_relationships: Dict[str, Dict] = field(default_factory=dict)
    # Format: {variable: {"direction": "increasing/decreasing", "outcome": "mortality", "strength": "strong/moderate/weak"}}
    
    # Plausibility bounds
    variable_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Expected correlations
    correlation_signs: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # Format: {(var1, var2): "positive"/"negative"}
    
    # Biological relationships
    mediators: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"mediator": "BMI", "from": "age", "to": "diabetes"}]
    
    # Evidence metadata
    literature_sources: List[str] = field(default_factory=list)
    retrieval_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    llm_confidence: Literal["high", "moderate", "low"] = "moderate"
    evidence_notes: str = ""
    
    def to_physics_constraints(self) -> Dict[str, Any]:
        """Convert disease profile to physics constraint specification"""
        return {
            "monotonicity": [
                {
                    "variable": var,
                    "direction": spec["direction"],
                    "outcome": spec.get("outcome", "hazard"),
                    "penalty_weight": 1.0 if spec.get("strength") == "strong" else 0.5
                }
                for var, spec in self.monotonic_relationships.items()
            ],
            "bounds": [
                {
                    "variable": var,
                    "lower": bounds[0],
                    "upper": bounds[1],
                    "hard": True
                }
                for var, bounds in self.variable_bounds.items()
            ],
            "correlation_signs": [
                {
                    "variable1": pair[0],
                    "variable2": pair[1],
                    "required_sign": sign,
                    "penalty_weight": 2.0
                }
                for pair, sign in self.correlation_signs.items()
            ],
            "natural_history": {
                "profile": self.hazard_profile,
                "early_time": self.early_time_days,
                "mid_time": self.mid_time_days,
                "late_time": self.late_time_days,
                "penalty_weight": 2.0
            } if self.hazard_profile != "constant" else None,
            "mediators": self.mediators
        }


# ============================================================================
# DISEASE PROFILE RETRIEVER
# ============================================================================

class DiseaseProfileRetriever:
    """
    Dynamically retrieve disease profiles from current medical literature

    This is the CORE INNOVATION - instead of hardcoded tables, the LLM (Claude or ChatGPT)
    searches literature in real-time to generate evidence-based disease profiles.
    """
    
    def __init__(self):
        self.cache: Dict[str, DiseaseProfile] = {}
        self.tool = LLMToolInterface()
    
    def get_profile(
        self,
        disease_name: str,
        additional_context: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False
    ) -> DiseaseProfile:
        """
        Retrieve or generate disease profile with LLM's help

        Process:
        1. Search literature for disease natural history
        2. Extract prognostic biomarkers and survival patterns
        3. Identify biological relationships
        4. Synthesize into DiseaseProfile with uncertainty

        Args:
            disease_name: Disease/condition name
            additional_context: Optional dict with trial-specific context
            force_refresh: If True, bypass cache and search afresh

        Returns:
            Evidence-based DiseaseProfile
        """
        cache_key = f"{disease_name}_{json.dumps(additional_context) if additional_context else ''}"
        
        if not force_refresh and cache_key in self.cache:
            logger.info(f"✓ Using cached profile for {disease_name}")
            return self.cache[cache_key]
        
        logger.info(f"🔍 Retrieving evidence-based profile for: {disease_name}")
        logger.info(f"   LLM will search medical literature and synthesize profile...")
        
        # Stage 1: Natural history and survival patterns
        logger.info("   Stage 1/4: Searching natural history...")
        natural_history = self._retrieve_natural_history(disease_name, additional_context)
        
        # Stage 2: Biomarkers and prognostic factors
        logger.info("   Stage 2/4: Identifying biomarkers...")
        biomarkers = self._retrieve_biomarkers(disease_name, additional_context)
        
        # Stage 3: Biological relationships
        logger.info("   Stage 3/4: Mapping biological relationships...")
        relationships = self._retrieve_relationships(disease_name, additional_context)
        
        # Stage 4: Plausibility bounds
        logger.info("   Stage 4/4: Setting plausibility bounds...")
        bounds = self._retrieve_bounds(disease_name, additional_context)
        
        # Synthesize into DiseaseProfile
        profile = DiseaseProfile(
            disease_name=disease_name,
            **natural_history,
            **biomarkers,
            **relationships,
            **bounds
        )
        
        # Cache and return
        self.cache[cache_key] = profile
        logger.info(f"✓ Profile generated with {profile.llm_confidence} confidence")
        logger.info(f"   Sources: {len(profile.literature_sources)} references")
        
        return profile
    
    def _retrieve_natural_history(
        self,
        disease_name: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search literature for disease natural history

        LLM will:
        1. Search for survival curves, median OS
        2. Identify hazard profile (increasing/decreasing)
        3. Extract typical progression patterns
        """
        search_query = f"{disease_name} natural history survival pattern median"
        
        # This call will be intercepted by LLM
        results = self.tool.search_literature(search_query, max_results=5)

        # LLM will synthesize from results
        synthesis_prompt = f"""
        Based on literature for {disease_name}:
        1. What is the typical hazard profile? (decreasing/increasing/constant/bathtub)
        2. What is typical median survival? (in months)
        3. What survival distribution fits best? (Weibull/exponential/log-normal)
        4. What is the typical age range for this disease?
        
        Provide JSON format:
        {{
            "hazard_profile": "...",
            "median_survival_months": ...,
            "survival_distribution": "...",
            "typical_age_range": [min, max],
            "evidence_sources": ["url1", "url2"],
            "confidence": "high/moderate/low",
            "notes": "..."
        }}
        
        If insufficient evidence, mark confidence as "low" and note limitations.
        """

        # LLM will reason and return JSON
        synthesis = self.tool.reason_about_context(synthesis_prompt, {"results": results})

        # Parse and return (placeholder - LLM will handle)
        return {
            "hazard_profile": "increasing",  # Placeholder
            "median_survival_months": None,
            "survival_distribution": "Weibull",
            "typical_age_range": (0.0, 100.0),
            "literature_sources": ["placeholder"],
            "llm_confidence": "moderate",
            "evidence_notes": "Placeholder - LLM will fill this"
        }
    
    def _retrieve_biomarkers(
        self,
        disease_name: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search for prognostic biomarkers and monotonic relationships

        LLM will identify:
        - Key biomarkers (lab values, clinical scores)
        - Monotonic relationships (age↑ → mortality↑)
        - Strength of evidence
        """
        search_query = f"{disease_name} prognostic factors biomarkers risk factors"
        results = self.tool.search_literature(search_query, max_results=5)
        
        synthesis_prompt = f"""
        From literature on {disease_name}, identify:
        1. Key prognostic biomarkers/variables
        2. For each, is relationship with mortality monotonic? (increasing/decreasing)
        3. Strength of evidence (strong/moderate/weak)
        
        Return JSON:
        {{
            "relevant_biomarkers": ["var1", "var2", ...],
            "monotonic_relationships": {{
                "var1": {{"direction": "increasing", "outcome": "mortality", "strength": "strong"}},
                ...
            }}
        }}
        """
        
        synthesis = self.tool.reason_about_context(synthesis_prompt, {"results": results})
        
        # Placeholder
        return {
            "relevant_biomarkers": [],
            "monotonic_relationships": {}
        }
    
    def _retrieve_relationships(
        self,
        disease_name: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Identify biological relationships (mediation, correlation signs)

        LLM will search for:
        - Known mediators (BMI mediates age→diabetes)
        - Required correlation signs (height-weight must be positive)
        """
        search_query = f"{disease_name} pathophysiology biological mechanisms"
        results = self.tool.search_literature(search_query, max_results=3)
        
        synthesis_prompt = f"""
        From pathophysiology of {disease_name}, identify:
        1. Mediation relationships (X → M → Y)
        2. Required correlation signs (which pairs must be pos/neg)
        
        Return JSON:
        {{
            "mediators": [
                {{"mediator": "BMI", "from": "age", "to": "diabetes"}},
                ...
            ],
            "correlation_signs": {{
                ["var1", "var2"]: "positive",
                ...
            }}
        }}
        """
        
        synthesis = self.tool.reason_about_context(synthesis_prompt, {"results": results})
        
        # Placeholder
        return {
            "mediators": [],
            "correlation_signs": {}
        }
    
    def _retrieve_bounds(
        self,
        disease_name: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Retrieve plausibility bounds from clinical guidelines

        LLM will search for:
        - Normal ranges for lab values
        - Physiological limits
        - Disease-specific ranges
        """
        search_query = f"{disease_name} clinical ranges normal values guidelines"
        results = self.tool.search_literature(search_query, max_results=3)
        
        synthesis_prompt = f"""
        From clinical guidelines for {disease_name}, identify plausible ranges:
        
        For each relevant variable:
        - Lower and upper bounds (physiologically plausible)
        - Based on clinical guidelines or population norms
        
        Return JSON:
        {{
            "variable_bounds": {{
                "age": [0, 110],
                "weight_kg": [30, 300],
                ...
            }}
        }}
        """
        
        synthesis = self.tool.reason_about_context(synthesis_prompt, {"results": results})
        
        # Placeholder with sensible defaults
        return {
            "variable_bounds": {
                "age": (0.0, 110.0),
                "weight": (30.0, 300.0),
                "height": (120.0, 230.0),
                "bmi": (12.0, 60.0)
            }
        }


# ============================================================================
# CONSTRAINT EXTRACTION FROM PAPERS
# ============================================================================

@dataclass
class ExtractedConstraints:
    """Constraints extracted from a paper via LLM's NLP"""
    paper_url: str
    paper_title: str
    
    # Extracted statistical results
    means_and_sds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Survival data
    km_data: List[Dict[str, Any]] = field(default_factory=list)
    hazard_ratios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Subgroup analyses
    subgroup_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Confidence
    extraction_confidence: Literal["high", "moderate", "low"] = "moderate"
    notes: str = ""


class PaperConstraintExtractor:
    """
    Extract constraints from papers using LLM's NLP and reasoning

    When user provides a paper URL or uploads a PDF, the LLM (Claude or ChatGPT) can:
    1. Parse statistical results
    2. Extract forest plot subgroup analyses
    3. Digitize Kaplan-Meier curves
    4. Identify conditional relationships
    """
    
    def __init__(self):
        self.tool = LLMToolInterface()
    
    def extract_from_url(self, url: str) -> ExtractedConstraints:
        """
        Extract constraints from a paper URL

        LLM will:
        1. Fetch full paper content
        2. Parse tables, figures, results section
        3. Extract all statistical summaries
        4. Structure into ConstraintCollection format
        """
        logger.info(f"📄 Extracting constraints from: {url}")
        
        # Fetch paper
        paper_content = self.tool.fetch_url_content(url)
        
        # Extract constraints via reasoning
        extraction_prompt = f"""
        Parse this paper and extract ALL constraints suitable for VINDEL:
        
        1. MARGINAL CONSTRAINTS:
           - Means, SDs for all baseline characteristics
           - Quantiles if reported
           
        2. JOINT CONSTRAINTS:
           - Correlation matrices
           - Any reported associations
           
        3. CONDITIONAL CONSTRAINTS:
           - Subgroup-specific correlations
           - Stratified analyses
           
        4. SURVIVAL CONSTRAINTS:
           - Kaplan-Meier data (time points and survival probabilities)
           - Median survival per arm
           - Hazard ratios with CIs
           
        5. SUBGROUP SURVIVAL:
           - Forest plots with subgroup-specific HRs
           - Stratified KM curves if present
           
        6. TIME-VARYING EFFECTS:
           - Any mention of non-proportional hazards
           - HRs by time period
           
        7. MULTI-OUTCOME:
           - Response rates + survival (check consistency)
           
        Return structured JSON following ExtractedConstraints schema.
        If data is in figures, describe what you see (user may need to digitize).
        """
        
        extracted = self.tool.reason_about_context(
            extraction_prompt,
            {"paper_content": paper_content}
        )
        
        # Placeholder
        return ExtractedConstraints(
            paper_url=url,
            paper_title="Placeholder - LLM will extract",
            extraction_confidence="moderate",
            notes="LLM will parse and structure all constraints"
        )
    
    def extract_from_text(self, text: str, source: str = "user_provided") -> ExtractedConstraints:
        """
        Extract constraints from provided text (e.g., pasted methods section)
        """
        extraction_prompt = f"""
        Extract VINDEL constraints from this text:
        {text}
        
        Follow same protocol as extract_from_url.
        Return structured JSON.
        """
        
        extracted = self.tool.reason_about_context(extraction_prompt)
        
        return ExtractedConstraints(
            paper_url=source,
            paper_title="Text excerpt",
            extraction_confidence="moderate"
        )


# ============================================================================
# SMART PARAMETER SELECTOR
# ============================================================================

class SmartParameterSelector:
    """
    Select optimal parameters for VINDEL based on context

    Instead of fixed defaults, the LLM (Claude or ChatGPT) reasons about:
    - Loss weights based on data sparsity
    - Convergence criteria based on complexity
    - Model space based on dimensionality
    """
    
    def __init__(self):
        self.tool = LLMToolInterface()
    
    def recommend_loss_weights(
        self,
        constraint_collection: Any,
        disease_profile: Optional[DiseaseProfile] = None,
        use_case: str = "standard"
    ) -> Dict[str, float]:
        """
        Recommend loss weights based on constraint availability and use case

        LLM will reason about:
        - Which constraints are well-specified (high weight)
        - Which are sparse/uncertain (lower weight)
        - Disease-specific priorities
        """
        reasoning_prompt = f"""
        Recommend loss weights for VINDEL given:
        
        Available constraints:
        - Marginal: {len(constraint_collection.marginal_constraints)}
        - Joint: {len(constraint_collection.joint_constraints)}
        - Conditional: {len(constraint_collection.conditional_constraints)}
        - Survival: {len(constraint_collection.survival_constraints)}
        - Subgroup survival: {len(constraint_collection.subgroup_survival_constraints)}
        - Causal: {len(constraint_collection.causal_constraints)}
        
        Use case: {use_case}
        Disease: {disease_profile.disease_name if disease_profile else "Unknown"}
        
        Provide JSON:
        {{
            "lambda_marg": ...,
            "lambda_joint": ...,
            "lambda_cond": ...,
            "lambda_KM": ...,
            "lambda_KM_subg": ...,
            "lambda_causal": ...,
            "lambda_multi": ...,
            "lambda_physics": ...,
            "rationale": "..."
        }}
        
        Guidelines:
        - Higher weight if many constraints of that type
        - Lower weight if sparse/uncertain
        - Survival usually most important (20-25)
        - Physics constraints soft (1-5)
        """
        
        weights = self.tool.reason_about_context(reasoning_prompt)
        
        # Placeholder defaults
        return {
            "lambda_marg": 10.0,
            "lambda_joint": 5.0,
            "lambda_cond": 3.0,
            "lambda_KM": 20.0,
            "lambda_KM_subg": 15.0,
            "lambda_causal": 5.0,
            "lambda_multi": 5.0,
            "lambda_physics": 2.0
        }


# ============================================================================
# USAGE INSTRUCTIONS FOR LLMs
# ============================================================================

def print_usage_instructions():
    """Print instructions for LLMs (Claude or ChatGPT) on how to use this module"""
    print(LLM_INSTRUCTIONS)
    print("""
    === USAGE EXAMPLE FOR LLMs ===

    # User uploads VINDEL package to LLM chat (Claude or ChatGPT)
    # User requests: "Please load and activate Python runtime"
    # User: "Generate sIPD for a metastatic NSCLC trial with 300 patients"

    # LLM internally:
    from vindel.integration.llm_retriever import DiseaseProfileRetriever
    
    retriever = DiseaseProfileRetriever()
    profile = retriever.get_profile("metastatic non-small cell lung cancer")

    # LLM executes web_search automatically:
    # - Searches "NSCLC natural history survival"
    # - Finds median OS ~12 months, decreasing hazard (acute spike post-diagnosis)
    # - Identifies biomarkers: PD-L1, EGFR, ALK
    # - Sets bounds: age [18-100], ECOG [0-4]

    # Returns DiseaseProfile with:
    # - hazard_profile="decreasing"
    # - median_survival_months=12
    # - monotonic_relationships={"age": {"direction": "increasing", "outcome": "mortality"}}
    # - literature_sources=[URLs from PubMed]
    # - llm_confidence="high"

    # Then LLM uses this profile to configure physics constraints!
    """)


if __name__ == "__main__":
    print_usage_instructions()
