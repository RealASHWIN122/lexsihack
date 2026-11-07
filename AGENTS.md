# AI Agent & Contributor Guidelines (`AGENTS.md`)

This repository is dedicated to mechanistic interpretability research, circuit discovery, model pruning, and red-teaming using the **CircuitKIT** framework.

All AI agents (Antigravity, Claude, Cursor, Copilot, ChatGPT, etc.) and human contributors working in this codebase **MUST STRICTLY ADHERE** to the following structural rules and architectural protocols.

---

## 1. Repository Architecture

```text
lexsihack/
├── AGENTS.md                          # Mandatory instructions for all AI agents
├── .gitignore                         # Git exclusion rules (venv, large safetensors, cache)
│
├── CircuitKIT/                        # Core library and engine
│   ├── src/circuitkit/                # Framework source code (YAML loaders, discovery, pruning)
│   ├── docs/                          # Official framework documentation
│   ├── tests/                         # Unit and integration test suites
│   ├── pyproject.toml                 # Packaging specification
│   └── .venv/                         # Shared Python virtual environment
│
├── experiments/                       # ALL experiments MUST reside here
│   ├── README.md                      # Experiments index and usage instructions
│   ├── _template/                     # Official blueprint for creating new experiments
│   │   ├── README.md                  # Instructions for replicating the template
│   │   ├── data/                      # Prompt pairs (data.csv) & task config (task.yaml)
│   │   ├── results/                   # Output artifacts (ignored/committed selectively)
│   │   └── run_experiment.py          # Self-contained discovery and evaluation pipeline
│   ├── experiment_01_capitals/        # Capital cities factual association task
│   └── <your_new_experiment>/         # Each contributor/agent creates a new folder here
│
├── aya_redteam_robustness/            # Red-teaming and safety evaluation tasks
└── research/                          # Reference papers, PDFs, and literature
```

---

## 2. Mandatory Rules for Agents

### Rule 1: Zero Clutter in Root and Library Directories
- **NEVER** create experiment scripts (`run_*.py`, `test_*.py`), scratch notebooks (`*.ipynb`), CSV datasets, or output files in the root folder (`/`) or inside `CircuitKIT/`.
- Root folder is strictly reserved for workspace-level configuration (`AGENTS.md`, `.gitignore`, top-level docs).
- `CircuitKIT/` is strictly reserved for the core framework source code and its internal test suite.

### Rule 2: Strict Isolation of Experiments
- **Every single experiment or task study MUST have its own dedicated subfolder inside `experiments/`**.
  - Folder naming convention: `experiments/experiment_<NN>_<descriptive_name>/` or `experiments/<topic>_<subtopic>/`
  - Example: `experiments/experiment_02_indirect_object_identification/`
  - Example: `experiments/aya_toxicity_circuits/`
- **Never share or overwrite another experiment's folder**. Keep each experiment fully independent and reproducible.

### Rule 3: Follow the Experiment Blueprint
When asked to perform a new experiment, analyze a circuit, or test a new prompt dataset:
1. Copy the blueprint from [`experiments/_template/`](file:///home/me/Videos/lexsihack/experiments/_template):
   ```bash
   cp -r experiments/_template experiments/experiment_<NN>_<name>
   ```
2. Place contrastive datasets inside `experiments/<name>/data/`.
3. Point task schemas inside `experiments/<name>/data/task.yaml`.
4. Ensure script paths resolve relative to `Path(__file__).resolve().parent` so they can run from any working directory.
5. Save all output circuits (`.pt`), faithfulness scores (`.json`), and interactive D3 graphs (`.html`) into `experiments/<name>/results/`.
6. Add an experiment `README.md` detailing:
   - Biological / mechanistic hypothesis.
   - Contrastive dataset schema (clean vs corrupted prompts).
   - Target model architecture (e.g. `gpt2`, `CohereForAI/c4ai-command-r-v01`).
   - Summary of discovered critical heads, MLPs, or token positions.

### Rule 4: Python Environment & Import Order
- Use the shared Python environment located at [`CircuitKIT/.venv`](file:///home/me/Videos/lexsihack/CircuitKIT/.venv).
- `circuitkit` is installed in editable mode (`pip install -e CircuitKIT`).
- **Critical import constraint**: Always import `circuitkit` **before** importing `transformer_lens`:
  ```python
  import circuitkit as ck
  import transformer_lens
  ```

### Rule 5: Large File & Checkpoint Discipline
- Discovered circuit files (`.pt`, `.json`, `.html`) are typically lightweight (< 1MB) and can be committed.
- **NEVER commit model checkpoints or weights exceeding 50MB** (e.g. `model.safetensors`, `pytorch_model.bin`). Ensure any checkpoint directories are gitignored.

---

## 3. Quick Checklist for Any Agent Starting a Task

1. [ ] Am I creating files inside `experiments/<my_experiment_name>/`?
2. [ ] Is the data isolated in `<my_experiment_name>/data/`?
3. [ ] Are results saving into `<my_experiment_name>/results/`?
4. [ ] Did I write a concise `README.md` inside my experiment folder?
5. [ ] Did I leave the repository root and `CircuitKIT/` clean?
