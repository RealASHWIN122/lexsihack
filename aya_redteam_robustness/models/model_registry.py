"""
Model loading, management, and inference interface for the Cohere Aya model family.
Supports:
1. Local Hugging Face / CircuitKIT checkpoint inference (bfloat16/float16).
2. Cohere v2 Chat API integration when COHERE_API_KEY is available.
3. Deterministic calibrated simulation mode ONLY when force_mock=True.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from aya_redteam_robustness.config import ALLOWED_MODELS
from aya_redteam_robustness.models.mock_engine import (
    simulate_response,
    TELEMETRY_REFUSAL_STRING
)


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face authentication token from environment or cache."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def get_cohere_api_key() -> Optional[str]:
    """Retrieve Cohere API key from environment."""
    return os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")


class AyaModelWrapper:
    """Unified evaluation wrapper around the Cohere Labs Aya models."""

    def __init__(
        self,
        model_key: str,
        use_mock: bool = False,
        prefer_api: bool = False,
        device: Optional[str] = None
    ):
        if model_key not in ALLOWED_MODELS:
            raise ValueError(f"Model '{model_key}' is not in allowed models: {list(ALLOWED_MODELS.keys())}")

        self.model_key = model_key
        self.info = ALLOWED_MODELS[model_key]
        self.repo_id = self.info["repo_id"]
        self.use_mock = use_mock
        self.prefer_api = prefer_api
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.cohere_client = None
        self.is_abliterated = False
        self.init_error = None

        if not use_mock:
            # 1. Check for Cohere API mode first if preferred
            api_key = get_cohere_api_key()
            if prefer_api and api_key:
                if not self._init_cohere_api(api_key):
                    raise RuntimeError(f"Failed to initialize Cohere API: {self.init_error}")
            else:
                # 2. Attempt local checkpoint loading
                loaded = self._load_live_model()
                if not loaded:
                    # 3. Fallback to API if available
                    if api_key and self._init_cohere_api(api_key):
                        pass
                    else:
                        raise RuntimeError(
                            f"Live model loading failed for '{self.repo_id}'.\n"
                            f"Cause: {self.init_error}\n"
                            f"Access to '{self.repo_id}' is gated on Hugging Face (https://huggingface.co/{self.repo_id}).\n"
                            f"Please provide an HF_TOKEN with accepted terms or set COHERE_API_KEY."
                        )

    def _init_cohere_api(self, api_key: str) -> bool:
        """Initialize Cohere v2 client."""
        try:
            import cohere
            self.cohere_client = cohere.ClientV2(api_key=api_key)
            self.use_mock = False
            return True
        except Exception as e:
            self.init_error = str(e)
            return False

    def _load_live_model(self) -> bool:
        """Attempt to load the model via transformers."""
        token = get_hf_token()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                token=token,
                trust_remote_code=True
            )
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                token=token,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to("cpu")
            self.use_mock = False
            return True
        except Exception as e:
            self.init_error = str(e)
            return False

    def generate(
        self,
        prompt: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """
        Generate response using active backend:
        1. Cohere API if initialized.
        2. Local PyTorch/Transformers model if loaded.
        3. Simulation engine ONLY if use_mock is explicitly True.
        """
        # 1. Cohere v2 API execution
        if self.cohere_client is not None and not self.use_mock:
            cohere_model_name = "command-r-08-2024" if "expanse" in self.model_key else "c4ai-aya-expanse-8b"
            response = self.cohere_client.chat(
                model=cohere_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            if response.message and response.message.content:
                return response.message.content[0].text.strip()
            return ""

        # 2. Local PyTorch model execution
        if not self.use_mock and self.model is not None and self.tokenizer is not None:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in encoded.items() if k != "token_type_ids"}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 3. Deterministic calibrated simulation (ONLY when explicitly requested)
        if self.use_mock:
            return simulate_response(
                model_key=self.model_key,
                prompt=prompt,
                item_metadata=item_metadata,
                is_abliterated=self.is_abliterated,
                temperature=temperature
            )

        raise RuntimeError("No valid execution backend initialized.")


def get_model(
    model_key: str,
    force_mock: bool = False,
    prefer_api: bool = False
) -> AyaModelWrapper:
    """Helper factory to retrieve model instance."""
    return AyaModelWrapper(model_key=model_key, use_mock=force_mock, prefer_api=prefer_api)
