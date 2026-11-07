# Experiment 01: World Capitals Task

This experiment studies factual association circuits in language models (specifically discovering the subnetwork in `gpt2` responsible for predicting capital cities given countries).

## Structure

- **`run_experiment.py`**: Pipeline script that loads the task, runs circuit discovery (EAP-IG), evaluates faithfulness, prunes the model, and exports the checkpoint.
- **`run_experiment.ipynb`**: Interactive Jupyter Notebook version of the experiment.
- **`circuitkit_app.py`**: Streamlit / visual application for exploring circuits.
- **`data/`**:
  - `data.csv`: Contrastive prompt pairs (e.g., `"The capital of France is"` -> `" Paris"` vs. counterfactual `"The capital of Italy is"` -> `" Rome"`).
  - `task.yaml`: CircuitKIT task specification schema.
- **`results/`**:
  - `circuit_graph.html`: Interactive D3 graph visualization of discovered circuit nodes.
  - `custom_circuit.pt`: Serialized discovered circuit tensor.
  - `custom_circuit_scores.json` / `custom_circuit_scores.pt`: Attribution and importance scores for circuit nodes.
  - `circuit_eval_results.json`: Faithfulness evaluation scores (causal patching and ablation recovery).
  - `checkpoint/`: Pruned model checkpoint with `lexsi_provenance.json`.

## How to Run

From anywhere in the repository:
```bash
/home/me/Videos/lexsihack/CircuitKIT/.venv/bin/python run_experiment.py
```
Or if you activated the virtual environment:
```bash
source /home/me/Videos/lexsihack/CircuitKIT/.venv/bin/activate
python run_experiment.py
```
