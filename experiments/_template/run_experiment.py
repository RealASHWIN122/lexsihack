#!/usr/bin/env python3
"""
Template Circuit Discovery Experiment Script using CircuitKIT.

To run:
    python run_experiment.py
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import torch

# Directory setup
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Always import circuitkit first before transformer_lens
import circuitkit as ck
from circuitkit.tasks.yaml_loader import YAMLTaskLoader
from circuitkit.tasks.registry import register_task


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
TASK_NAME = "my_custom_task"
MODEL_NAME = "gpt2"          # e.g., 'gpt2', 'gpt2-medium', etc.
ALGORITHM = "eap-ig"         # e.g., 'eap-ig', 'act-add', 'eap'
SPARSITY = 0.2               # Keep top 80% nodes, prune lowest 20%


def load_or_create_task(name: str) -> str:
    """Load or register declarative task specification from data/task.yaml."""
    yaml_file = DATA_DIR / "task.yaml"
    csv_file = DATA_DIR / "data.csv"

    if not yaml_file.exists():
        raise FileNotFoundError(f"Missing task configuration: {yaml_file}")

    task_spec = YAMLTaskLoader.load(yaml_file)
    register_task(task_spec, overwrite=True)
    print(f"✓ Task '{name}' loaded and registered in CircuitKIT.")
    return name


def main():
    print("=" * 60)
    print(f"Starting Experiment: {TASK_NAME}")
    print("=" * 60)

    # 1. Register / Load Task
    task_id = load_or_create_task(TASK_NAME)

    # 2. Load model
    print(f"\n[Step 1] Loading model '{MODEL_NAME}'...")
    model = ck.load_model(MODEL_NAME)

    # 3. Discover circuit
    print(f"\n[Step 2] Discovering circuit using {ALGORITHM}...")
    circuit_out = RESULTS_DIR / "circuit.pt"
    circuit = ck.discover(
        model,
        task=task_id,
        algorithm=ALGORITHM,
        level="node",
        sparsity=SPARSITY,
        output_path=circuit_out
    )
    print(f"✓ Discovered circuit containing {len(circuit)} nodes.")

    # 4. Faithfulness Evaluation
    print("\n[Step 3] Evaluating Faithfulness...")
    report = ck.faithfulness(
        model,
        circuit,
        task=task_id,
        pillars=["patching", "ablation"]
    )
    print(f"  • Causal Patching Score: {report.patching_score:.4f}")
    print(f"  • Ablation Score:        {report.ablation_score:.4f}")

    # 5. Interactive D3 Visualization
    viz_file = RESULTS_DIR / "circuit_graph.html"
    print(f"\n[Step 4] Saving visualization to {viz_file}...")
    ck.visualize_circuit(circuit, mode="graph", output=str(viz_file))

    # 6. Optional: Prune & Export
    print("\n[Step 5] Pruning model & Exporting checkpoint...")
    pruned_model = ck.prune(model, circuit, sparsity=SPARSITY, scope="both")
    export_dir = RESULTS_DIR / "checkpoint"
    ck.export_checkpoint(pruned_model, circuit, str(export_dir))

    print(f"\n✓ Experiment complete! Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
