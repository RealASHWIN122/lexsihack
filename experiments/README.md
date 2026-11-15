# Experiments

This folder contains all custom experiments conducted with **CircuitKIT**. Each experiment is self-contained in its own folder with its own datasets, scripts, configurations, and outputs.

## Directory Overview

```text
experiments/
├── README.md                                  # This file
├── _template/                                 # Blueprint for new experiments
│   ├── README.md
│   ├── data/
│   │   ├── data.csv
│   │   └── task.yaml
│   └── run_experiment.py
│
├── experiment_01_capitals/                    # Capital cities circuit discovery task
│   ├── README.md
│   ├── run_experiment.py
│   ├── run_experiment.ipynb
│   ├── circuitkit_app.py
│   ├── data/
│   │   ├── capitals.csv
│   │   └── task.yaml
│   └── results/
│       ├── circuit_graph.html
│       ├── custom_circuit.pt
│       ├── custom_circuit_scores.json
│       ├── circuit_eval_results.json
│       └── checkpoint/
│
├── experiment_02_aya_fire_lexsistack/         # Targeted Model Repair on CohereLabs/tiny-aya-fire
│   ├── README.md
│   ├── run_experiment.py
│   ├── app.py
│   ├── data/
│   │   ├── data.csv
│   │   └── task.yaml
│   └── results/
│       ├── audit_ledger.json
│       ├── circuit.pt
│       ├── circuit_graph.html
│       └── experiment_summary.json
│
└── experiment_03/                             # Multilingual Safety Benchmark & Targeted Model Repair
    ├── README.md
    ├── app.py
    ├── run_experiment.py
    ├── run_pair04_attribution.py
    ├── apply_circuit_repair.py
    ├── data/
    │   ├── aya_safecal_bench.jsonl
    │   ├── data.csv
    │   ├── DATASET_CARD.md
    │   └── task.yaml
    ├── reports/
    │   └── AYA_SAFETY_BENCHMARK_REPORT.md
    └── results/
        ├── audit_ledger.json
        ├── aya_raw_evaluations.jsonl
        ├── aya_summary_metrics.csv
        ├── aya_summary_metrics.json
        ├── circuit.pt
        ├── circuit_graph.html
        ├── circuit_nodes.json
        └── circuit_scores.json
```

## Quick Start: Starting a New Experiment

To start a new experiment:

```bash
cd experiments
cp -r _template experiment_<NN>_<task_name>
cd experiment_<NN>_<task_name>

# 1. Edit data/data.csv with your prompts
# 2. Run the experiment
python run_experiment.py
```
