"""
Configuration and model definitions for the Cohere Aya Safety and Robustness Benchmark.
Evaluates selective refusal calibration, false-positive over-refusal, and cross-regional consistency.
"""

from typing import Dict, List, Any

# Canonical list of 6 Cohere Labs models evaluated in the benchmark
ALLOWED_MODELS: Dict[str, Dict[str, Any]] = {
    "tiny-aya-base": {
        "repo_id": "CohereLabs/tiny-aya-base",
        "description": "3.35B pretrained foundation baseline without instruction or safety alignment (70 languages).",
        "params": "3.35B",
        "alignment_type": "None (Pretrained Foundation)",
        "specialization": "Foundation",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-global": {
        "repo_id": "CohereLabs/tiny-aya-global",
        "description": "3.35B balanced multilingual instruction-tuned model across global languages.",
        "params": "3.35B",
        "alignment_type": "Global Multilingual SFT + DPO",
        "specialization": "Global (70+ languages)",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-fire": {
        "repo_id": "CohereLabs/tiny-aya-fire",
        "description": "3.35B regional specialist optimized for South Asian languages (Hindi, Malayalam, Tamil, etc.).",
        "params": "3.35B",
        "alignment_type": "Regional Instruction Alignment (South Asia)",
        "specialization": "South Asian",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-water": {
        "repo_id": "CohereLabs/tiny-aya-water",
        "description": "3.35B regional specialist optimized for European and Asia-Pacific languages (Spanish, Indonesian, etc.).",
        "params": "3.35B",
        "alignment_type": "Regional Instruction Alignment (Europe & APAC)",
        "specialization": "European & APAC",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-earth": {
        "repo_id": "CohereLabs/tiny-aya-earth",
        "description": "3.35B regional specialist optimized for West Asian and African languages (Arabic, Swahili, etc.).",
        "params": "3.35B",
        "alignment_type": "Regional Instruction Alignment (MENA & Africa)",
        "specialization": "MENA & Africa",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "aya-expanse": {
        "repo_id": "CohereForAI/aya-expanse-8b",
        "description": "8B scaled multilingual reference checkpoint with comprehensive instruction and safety alignment.",
        "params": "8.0B",
        "alignment_type": "Full RLHF / DPO Alignment",
        "specialization": "Global Scaled Reference",
        "family": "Aya Expanse",
        "architecture": "cohere"
    }
}

EVALUATION_DOMAINS: List[str] = [
    "clinical_triage",
    "sysadmin_privacy",
    "financial_compliance"
]

CATEGORY_LABELS: List[str] = [
    "benign_technical",
    "contextual_dual_use",
    "policy_boundary"
]

REGIONAL_CLUSTERS: Dict[str, List[str]] = {
    "global": ["en"],
    "fire": ["hi", "ml", "ta"],
    "water": ["es", "id"],
    "earth": ["ar", "sw"]
}
