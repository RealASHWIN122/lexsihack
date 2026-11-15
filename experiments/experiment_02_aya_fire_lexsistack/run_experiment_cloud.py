#!/usr/bin/env python3
"""
Cloud-ready entrypoint for Experiment 02 (Lexsi Alignment & Safety Stack).

Leaves `run_experiment.py` untouched for local/collaborator runs, and adds:
  - Hugging Face Hub loading (`--model`, default: CohereLabs/tiny-aya-fire)
  - `HF_TOKEN` authentication (from env or `.env`)
  - Automatic GPU (`cuda` + `bfloat16`) selection to avoid CPU RAM OOMs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import run_experiment as local_exp  # noqa: E402
from circuitkit.tasks.registry import get_task, register_task  # noqa: E402
from circuitkit.tasks.yaml_loader import YAMLTaskLoader  # noqa: E402

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_hf_token(cli_token: str | None) -> str | None:
    """Resolve Hugging Face token from CLI flag or environment variables."""
    token = (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:
            print(f"Notice: huggingface_hub.login skipped ({exc}); using token directly.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud GPU runner for Lexsi Stack on HF Hub models")
    parser.add_argument(
        "--model",
        type=str,
        default="CohereLabs/tiny-aya-fire",
        help="Hugging Face repo ID or local model path",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype ('auto' uses bfloat16 on CUDA, float32 on CPU)",
    )
    parser.add_argument("--eval-n", type=int, default=6, help="Samples per benchmark suite (0 = full dataset)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens per generation")
    parser.add_argument("--alpha", type=float, default=1.0, help="C-ΔΘ steering strength")
    parser.add_argument("--judge-model", type=str, default=None, help="AuditKIT LLMJudge spec (e.g. openai:gpt-4o-mini)")
    parser.add_argument("--rejudge-only", action="store_true", help="Re-score cached results/aya_fire_generations.json")
    parser.add_argument("--keep-system-preamble", action="store_true", help="Keep Cohere's default safety preamble")
    args = parser.parse_args()

    hf_token = resolve_hf_token(args.hf_token)
    if not hf_token and not Path(args.model).exists():
        print("Warning: HF_TOKEN is not set. Gated Hugging Face models will fail to download.")

    use_cuda = torch.cuda.is_available()
    dtype_name = ("bfloat16" if use_cuda else "float32") if args.dtype == "auto" else args.dtype
    torch_dtype = DTYPE_MAP[dtype_name]

    # Point shared module constants to the requested HF model so provenance/reports record it accurately
    local_exp.MODEL_DIR = args.model

    print("*" * 80)
    print(f"  CLOUD LEXSI STACK RUNNER | Model: {args.model} | Device: {'cuda' if use_cuda else 'cpu'} ({dtype_name})")
    print("*" * 80)
    t_start = time.time()

    # 1. CuratorKIT
    contrast_rows = local_exp.run_curatorkit_pipeline(local_exp.DATA_CSV)

    # Load Task YAML
    task_spec = YAMLTaskLoader.load(local_exp.TASK_YAML)
    register_task(task_spec, overwrite=True)
    registered_task = get_task(task_spec.name)

    if args.rejudge_only and (local_exp.RESULTS_DIR / "aya_fire_generations.json").exists():
        model, tokenizer = None, None
        circuit_info = None
        repair_data = {}
        target_layers = []
    else:
        print(f"\nLoading {args.model} in {dtype_name} ({'CUDA' if use_cuda else 'CPU'})...")
        t_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {"dtype": torch_dtype, "low_cpu_mem_usage": True, "token": hf_token}
        if use_cuda:
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.eval()
        print(f"✓ Model loaded in {time.time() - t_load:.2f}s")

        # 2. CircuitKIT + SafeTune.interpret
        _, circuit_info, target_layers = local_exp.run_real_circuit_discovery(
            model, tokenizer, registered_task, contrast_rows
        )

        # 3. SafeTune.recover
        repair_data = local_exp.compute_repairs_via_safetune(
            model, tokenizer, contrast_rows, circuit_info, target_layers, alpha=args.alpha
        )

    # 4. AuditKIT + SafeTune.evaluate
    ledger = local_exp.run_real_evaluation(
        model=model,
        tokenizer=tokenizer,
        circuit_info=circuit_info,
        repair_data=repair_data,
        eval_n=args.eval_n,
        max_new_tokens=args.max_new_tokens,
        keep_system_preamble=args.keep_system_preamble,
        judge_model_spec=args.judge_model,
        rejudge_only=args.rejudge_only,
        alpha=args.alpha,
    )

    # 5. AlignTune & MANSU scope check
    local_exp.document_aligntune_and_mansu_roles()

    elapsed = round(time.time() - t_start, 2)
    summary = {
        "model": args.model,
        "dtype": dtype_name,
        "cuda": use_cuda,
        "target_layers": target_layers,
        "elapsed_seconds": elapsed,
        "llm_judge_used": ledger.get("llm_judge_used", False),
        "eval_n": args.eval_n,
    }
    (local_exp.RESULTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ CLOUD PIPELINE COMPLETED IN {elapsed}s")


if __name__ == "__main__":
    main()
