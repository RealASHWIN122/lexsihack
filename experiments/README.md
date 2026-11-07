# Experiments

This folder contains all custom experiments conducted with **CircuitKIT**. Each experiment is self-contained in its own folder with its own datasets, scripts, configurations, and outputs.

## Directory Overview

```text
experiments/
├── README.md                          # This file
├── _template/                         # Blueprint for new experiments
│   ├── README.md
│   ├── data/
│   │   ├── data.csv
│   │   └── task.yaml
│   └── run_experiment.py
│
├── experiment_01_capitals/            # Capital cities circuit discovery task
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
```

## Quick Start: Starting a New Experiment

To start a new experiment:

```bash
cd experiments
cp -r _template experiment_02_task_name
cd experiment_02_task_name

# 1. Edit data/data.csv with your prompts
# 2. Run the experiment
python run_experiment.py
```

## Python Environment

Activate the virtual environment located in `CircuitKIT`:

```bash
source /home/me/Videos/lexsihack/CircuitKIT/.venv/bin/activate
```
(CircuitKIT is installed in editable mode, so `import circuitkit` works everywhere.)
