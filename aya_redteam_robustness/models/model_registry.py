"""
Model loading, management, and inference interface for the Aya model family.
Supports both real Hugging Face model loading (with optional quantization/fallback)
and a calibrated high-fidelity mock engine for development, demo, and CPU machines.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from aya_redteam_robustness.config import ALLOWED_MODELS

# Check Hugging Face token
def get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


class AyaModelWrapper:
    """Unified wrapper around Aya models (live HuggingFace or Calibrated Simulator)."""

    def __init__(self, model_key: str, use_mock: bool = False, device: Optional[str] = None):
        if model_key not in ALLOWED_MODELS:
            raise ValueError(f"Model '{model_key}' is not in the allowed models list: {list(ALLOWED_MODELS.keys())}")
        
        self.model_key = model_key
        self.info = ALLOWED_MODELS[model_key]
        self.repo_id = self.info["repo_id"]
        self.use_mock = use_mock
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_abliterated = False

        if not use_mock:
            self._load_live_model()

    def _load_live_model(self):
        """Attempt to load the model via transformers."""
        token = get_hf_token()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading tokenizer for {self.repo_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                token=token,
                trust_remote_code=True
            )
            print(f"Loading weights for {self.repo_id} on {self.device}...")
            # Load in bfloat16 or float16 if cuda, else float32
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                token=token,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if not torch.cuda.is_available():
                self.model.to("cpu")
            self.use_mock = False
            print(f"Successfully loaded live model {self.repo_id}")
        except Exception as e:
            print(f"Warning: Could not load live model '{self.repo_id}' ({e}). Falling back to Calibrated Emulation Engine.")
            self.use_mock = True

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate response from model or mock engine."""
        if self.use_mock or self.model is None or self.tokenizer is None:
            from aya_redteam_robustness.models.mock_engine import simulate_response
            return simulate_response(
                model_key=self.model_key,
                prompt=prompt,
                is_abliterated=self.is_abliterated,
                temperature=temperature
            )

        # Real model inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # Decode only the generated continuation
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def get_model(model_key: str, force_mock: bool = False) -> AyaModelWrapper:
    """Helper factory to retrieve model instance."""
    return AyaModelWrapper(model_key=model_key, use_mock=force_mock)
