# LexsiHack: CircuitKIT Mechanistic Interpretability & Red-Teaming

This repository hosts mechanistic interpretability research, circuit discovery, model pruning, and red-teaming tasks built on top of [CircuitKIT](CircuitKIT/).

## Repository Layout

- **[`AGENTS.md`](AGENTS.md)**: **Mandatory rules and instructions for all AI agents and contributors** on how to structure experiments and avoid codebase clutter.
- **[`CircuitKIT/`](CircuitKIT/)**: The core engine and mechanistic interpretability library source code.
- **[`experiments/`](experiments/)**: The central directory for all custom experiments.
  - **[`experiments/_template/`](experiments/_template/)**: Template to clone when starting a new experiment.
  - **[`experiments/experiment_01_capitals/`](experiments/experiment_01_capitals/)**: Capital cities factual association circuit discovery.
- **[`aya_redteam_robustness/`](aya_redteam_robustness/)**: Red-teaming and safety evaluation tasks for Aya and related models.
- **[`research/`](research/)**: Reference literature, research papers, and technical reports.

## Getting Started

### 1. Set Up the Environment
The project uses a Python virtual environment located in `CircuitKIT/.venv`:

```bash
source CircuitKIT/.venv/bin/activate
pip install -e CircuitKIT
```

### 2. Creating a New Experiment
All contributors and AI agents must follow the protocol defined in [`AGENTS.md`](AGENTS.md):

```bash
cd experiments
cp -r _template experiment_<NN>_<name>
cd experiment_<NN>_<name>

# Edit data/data.csv with your contrastive prompt pairs
# Run the discovery pipeline
python run_experiment.py
```
