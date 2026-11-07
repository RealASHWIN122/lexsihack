# Experiment Template

Use this template as the starting point for every new experiment.

## How to Create a New Experiment

1. **Copy this folder**:
   ```bash
   cp -r _template experiment_02_your_task_name
   cd experiment_02_your_task_name
   ```

2. **Add your data**:
   - Edit `data/data.csv` with your contrastive prompt pairs:
     - `prompt`: The clean sentence (e.g. `"The capital of France is"`)
     - `answer`: Expected next token (e.g. `" Paris"`)
     - `corrupted_prompt`: Counterfactual prompt (e.g. `"The capital of Italy is"`)
     - `corrupted_answer`: Counterfactual distractor token (e.g. `" Rome"`)

3. **Configure the task** (optional):
   - Check `data/task.yaml` to ensure schema matches column names in `data/data.csv`.

4. **Run the experiment**:
   ```bash
   python run_experiment.py
   ```
   All circuit checkpoints, evaluation scores, and HTML visualizations will automatically be saved into `results/`.
