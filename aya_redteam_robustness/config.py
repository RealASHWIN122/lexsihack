"""
Configuration and model definitions for the LexSI HABS Hackathon:
Red-teaming and Robustness Evaluation of Aya Models.
"""

from typing import Dict, List, Any

# Canonical list of models allowed for the hackathon
ALLOWED_MODELS: Dict[str, Dict[str, Any]] = {
    "tiny-aya-base": {
        "repo_id": "CohereLabs/tiny-aya-base",
        "description": "Base multilingual foundation model (3.35B) without instruction or safety alignment.",
        "params": "3.35B",
        "alignment_type": "None (Raw Base)",
        "specialization": "Foundation",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-global": {
        "repo_id": "CohereLabs/tiny-aya-global",
        "description": "Globally balanced instruction & safety tuned model across 70+ languages.",
        "params": "3.35B",
        "alignment_type": "Global Multilingual SFT + DPO",
        "specialization": "Global (70+ languages)",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-fire": {
        "repo_id": "CohereLabs/tiny-aya-fire",
        "description": "Regionally specialized variant optimized for South Asian languages (Hindi, Urdu, Bengali, Tamil, etc.).",
        "params": "3.35B",
        "alignment_type": "Regional Instruction Alignment (South Asia)",
        "specialization": "South Asian",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-water": {
        "repo_id": "CohereLabs/tiny-aya-water",
        "description": "Regionally specialized variant optimized for European & Asia-Pacific languages (French, German, Chinese, Japanese, etc.).",
        "params": "3.35B",
        "alignment_type": "Regional Instruction Alignment (Europe & APAC)",
        "specialization": "European & APAC",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "tiny-aya-earth": {
        "repo_id": "CohereLabs/tiny-aya-earth",
        "description": "Regionally specialized variant optimized for West Asian & African languages (Arabic, Turkish, Persian, Swahili, etc.).",
        "params": "3.35B",
        "alignment_type": "Regional Instruction Alignment (MENA & Africa)",
        "specialization": "MENA & Africa",
        "family": "Tiny Aya",
        "architecture": "cohere2"
    },
    "aya-expanse-8b": {
        "repo_id": "CohereLabs/aya-expanse-8b",
        "description": "Scaled 8B multilingual safety-aligned instruction model.",
        "params": "8.0B",
        "alignment_type": "Full RLHF / DPO Alignment",
        "specialization": "Global Scaled",
        "family": "Aya Expanse",
        "architecture": "cohere"
    },
    "aya-expanse-32b": {
        "repo_id": "CohereLabs/aya-expanse-32b",
        "description": "Flagship 32B multilingual frontier instruction model.",
        "params": "32.3B",
        "alignment_type": "Full RLHF / DPO Alignment",
        "specialization": "Flagship Scaled",
        "family": "Aya Expanse",
        "architecture": "cohere"
    }
}

ATTACK_CATEGORIES: List[str] = [
    "Jailbreaks (Roleplay / Persona / Simulation)",
    "Prompt Injection (System Override / Delimiter Hijacking)",
    "Encoding Tricks (Base64, ROT13, Leetspeak, Multilingual)",
    "Best-of-N (BoN) Sampling Attack",
    "Weight-Space Attack (Refusal Direction Abliteration)"
]

HARM_CATEGORIES: List[str] = [
    "Cyber & Exploitation",
    "Chemical, Biological & Physical Safety",
    "Deception & Misinformation",
    "Fraud & Social Engineering",
    "Harassment & Hate"
]

BENCHMARK_MODES: List[str] = [
    "harmful_adversarial",  # Test Attack Success Rate (ASR)
    "over_refusal_benign",  # Test False Refusal Rate (FRR)
    "multilingual_transfer" # Test cross-lingual safety disparity
]
