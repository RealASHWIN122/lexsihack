#!/usr/bin/env python3
"""
Run Circuit Discovery, Evaluation & Pruning on YOUR OWN CUSTOM TEXT in CircuitKIT.

Usage:
    python run_custom_task.py
"""

import os
from pathlib import Path
import pandas as pd
import yaml
import torch

# 1. Always import circuitkit first before transformer_lens
import circuitkit as ck
from circuitkit.tasks.yaml_loader import YAMLTaskLoader
from circuitkit.tasks.registry import register_task, is_task_registered


# =============================================================================
# 1. DEFINE YOUR CUSTOM TEXT DATA HERE
# =============================================================================
# A task requires contrastive pairs:
# - prompt: The clean sentence
# - answer: The expected target token (note leading space if standard tokenizer)
# - corrupted_prompt: Counterfactual sentence with key entity/attribute modified
# - corrupted_answer: Distractor token produced by the counterfactual
# =============================================================================
MY_PROMPTS = [
    {
        "prompt": "The capital of France is",
        "answer": " Paris",
        "corrupted_prompt": "The capital of Italy is",
        "corrupted_answer": " Rome"
    },
    {
        "prompt": "The capital of Spain is",
        "answer": " Madrid",
        "corrupted_prompt": "The capital of Germany is",
        "corrupted_answer": " Berlin"
    },
    {
        "prompt": "The capital of Japan is",
        "answer": " Tokyo",
        "corrupted_prompt": "The capital of China is",
        "corrupted_answer": " Beijing"
    },
    {
        "prompt": "The capital of England is",
        "answer": " London",
        "corrupted_prompt": "The capital of Egypt is",
        "corrupted_answer": " Cairo"
    },
]

TASK_NAME = "custom_capitals"
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
OUTPUT_DIR = EXP_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HELPER: REGISTER CUSTOM TASK WITH CIRCUITKIT
# =============================================================================
def setup_custom_task(name: str, prompts: list) -> str:
    """Save dataset to CSV and register declarative task specification."""
    csv_file = DATA_DIR / "data.csv"
    yaml_file = DATA_DIR / "task.yaml"

    # Save prompt pairs to CSV
    df = pd.DataFrame(prompts)
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved {len(df)} custom prompt pairs to {csv_file}")

    # Create task schema YAML
    task_config = {
        "name": name,
        "source": {
            "type": "csv",
            "path": str(csv_file)
        },
        "schema": {
            "prompt": "prompt",
            "answer": "answer",
            "corrupted_prompt": "corrupted_prompt",
            "corrupted_answer": "corrupted_answer"
        },
        "metric": "logit_diff"
    }

    with open(yaml_file, "w") as f:
        yaml.dump(task_config, f, default_flow_style=False)

    # Register task dynamically in CircuitKIT (overwrite if already registered)
    task_spec = YAMLTaskLoader.load(yaml_file)
    register_task(task_spec, overwrite=True)
    print(f"✓ Task '{name}' registered in CircuitKIT.")
    return name


# =============================================================================
# 3. MAIN WORKFLOW: LOAD -> DISCOVER -> EVALUATE -> PRUNE -> EXPORT
# =============================================================================
def main():
    print("=" * 60)
    print("CircuitKIT: Running On User-Defined Custom Text")
    print("=" * 60)

    # 1. Register task
    task_id = setup_custom_task(TASK_NAME, MY_PROMPTS)

    # 2. Load model (e.g., 'gpt2' runs fast on CPU)
    print("\n[Step 1] Loading model 'gpt2'...")
    model = ck.load_model("gpt2")

    # 3. Discover circuit
    print(f"\n[Step 2] Discovering circuit for '{task_id}' using EAP-IG...")
    circuit = ck.discover(
        model,
        task=task_id,
        algorithm="eap-ig",       # Edge Attribution Patching with Integrated Gradients
        level="node",             # Identify critical Attention Heads and MLPs
        sparsity=0.2,             # Prune 20% lowest-scoring nodes (retain top 80%)
        n_examples=len(MY_PROMPTS),
        batch_size=2,
        ig_steps=2,
        output_path=OUTPUT_DIR / "custom_circuit.pt"
    )

    print(f"\n✓ Discovered circuit containing {len(circuit)} nodes.")
    print("Top 5 critical components for this behavior:")
    for comp, score in circuit.top_nodes(5).items():
        print(f"  • {comp:8s} : {score:.6f}")

    # 4. Score faithfulness (Causal Patching & Ablation)
    print("\n[Step 3] Evaluating Faithfulness...")
    report = ck.faithfulness(
        model,
        circuit,
        task=task_id,
        pillars=["patching", "ablation"],
        n_examples=len(MY_PROMPTS)
    )
    print(f"  • Causal Patching Faithfulness: {report.patching_score:.4f} (1.0 = perfect recovery)")
    print(f"  • Ablation Drop Score:         {report.ablation_score:.4f} (0.0 = complete drop when removed)")

    # 5. Generate interactive D3 visualization
    viz_file = OUTPUT_DIR / "circuit_graph.html"
    print(f"\n[Step 4] Generating interactive D3 circuit visualization → {viz_file}")
    ck.visualize_circuit(circuit, mode="graph", output=str(viz_file))
    print(f"✓ Interactive HTML saved to: {viz_file}")

    # 6. Structurally prune model
    print("\n[Step 5] Pruning model down to circuit...")
    pruned_model = ck.prune(model, circuit, sparsity=0.2, scope="both")

    # 7. Export HuggingFace checkpoint
    export_dir = OUTPUT_DIR / "checkpoint"
    print(f"\n[Step 6] Exporting HuggingFace checkpoint → {export_dir}")
    ckpt_path = ck.export_checkpoint(pruned_model, circuit, str(export_dir))
    print(f"✓ Checkpoint written to: {ckpt_path} (includes lexsi_provenance.json)")

    # 8. Test inference on custom text
    test_sentence = "The capital of France is"
    print(f"\n[Step 7] Testing model completion on: '{test_sentence}'")
    tokens = model.to_tokens(test_sentence)
    with torch.no_grad():
        logits = model(tokens)
    top_token = model.to_string(logits[0, -1].argmax())
    print(f"Predicted next token: {repr(top_token)}")
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
