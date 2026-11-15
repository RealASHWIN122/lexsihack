# Experiment 02: Targeted Model Repair on Aya-Fire (`CohereLabs/tiny-aya-fire`)

This experiment evaluates, diagnoses, and repairs multilingual and cybersecurity alignment vulnerabilities in **`CohereLabs/tiny-aya-fire`** (`Cohere2ForCausalLM`, 3.35B parameters, 36 layers) using the **Lexsi Alignment and Safety Stack** (**CuratorKIT**, **CircuitKIT**, **SafeTune**, and **AuditKIT**).

Two entrypoints are provided:
* **`run_experiment.py`** — Local runner expecting weights on disk at `/home/me/models/tiny-aya-fire` (`torch.float32`, CPU path).
* **`run_experiment_cloud.py`** — Cloud/GPU runner that pulls `CohereLabs/tiny-aya-fire` (or any HF repo / local path via `--model`) using `HF_TOKEN` and automatically uses `CUDA + bfloat16`.

---

## 1. Mechanistic Hypothesis & Target Model

* **Target Model**: `CohereLabs/tiny-aya-fire` (`Cohere2ForCausalLM`, 3.35B parameters, 36 transformer layers).
* **Mechanistic Hypothesis**: Refusal mechanisms in regionally specialized multilingual models are often entangled with English safety training, causing non-English harmful prompts (e.g., Hindi, Bengali, Tamil, Telugu, Kannada) and dual-use cybersecurity prompts to under-trigger late-layer refusal circuits.
* **Repair Strategy ($C\text{-}\Delta\Theta$)**: By localizing safety-critical MLP layers via **EAP-IG** (`CircuitKIT` + `SafeTune.interpret`) and applying bounded contrastive gradient updates strictly to `model.layers.<layer>.mlp.down_proj.weight` (`SafeTune.recover`), refusal behavior is strengthened across South Asian languages and offensive cybersecurity queries while leaving non-circuit weights (`embed_tokens`, `self_attn`, `gate_proj`, non-target layers) untouched (`max |ΔW_outside_circuit| == 0.0`).
* **Serving Overhead**: Zero extra parameters, zero runtime router/guardrail latency, and identical tokenizer/chat template.

---

## 2. Contrastive Dataset & Task Schema

### Contrastive Repair Dataset (`data/data.csv` & `data/task.yaml`)
* **Task config**: `data/task.yaml` (`name: aya_fire_safety_task`, `metric: logit_diff`).
* **CSV schema**: `lang,track,clean_prompt,clean_answer,corrupted_prompt,corrupted_answer`
  * **`multilingual_safety`** (6 contrastive pairs across `en`, `hi`, `bn`, `ta`, `te`, `kn`): pairs a benign defensive/educational query (`clean_prompt` → `clean_answer`) with a harmful counterpart (`corrupted_prompt` → refusal `corrupted_answer`).
  * **`cybersecurity_safety`** (2 contrastive pairs in `en`): pairs defensive security engineering (`Sysmon` detection rules, AWS `IMDSv2` hardening) with offensive exploitation requests (process hollowing/LSASS dumping, SSRF credential exfiltration).
* **Strict Disjointness**: Step 1 automatically checks every `clean_prompt` and `corrupted_prompt` against all held-out JSON benchmarks in `data/benchmarks/` and aborts if any overlap exists.

### Held-Out Benchmark Suites (`data/benchmarks/`)
Evaluated across all three checkpoints (**Base**, **Repair 1: Multilingual**, **Repair 2: Cybersecurity**):
* **Refusal (Harmful $\uparrow$)**: `harmbench_direct_200.json`, `strongreject_full.json`, `multijail_10lang_full.json` (`en`, `bn`, `ar`, `zh`), `indicsafe_full.json`, `banglasafe_full.json`, `fortress_public_full.json`, `rmcbench_full.json`, `cybersec_dualuse_full.json` (offensive subset).
* **Over-Refusal / Usability (Benign $\downarrow$)**: `or_bench_hard_1k.json`, `xstest_full.json` (`is_safe`), `jbb_benign_full.json`.
* **Defensive Utility & Boundary Grading (`E/O/C/R/F`)**: `cybersec_defensive_10_tasks.json` (10 defensive tasks + boundary ARP spoofing prompt).
* **Hazardous Knowledge Retention ($\leftrightarrow$)**: `wmdp_cyber_full.json`, `wmdp_bio_full.json`, `wmdp_chem_full.json` (4-choice A/B/C/D log-likelihood MCQ).

*(To rebuild or refresh the open-access benchmark JSONs from Hugging Face, run `data/build_full_datasets.py`.)*

---

## 3. End-to-End 5-Stage Pipeline Architecture

Both `run_experiment.py` and `run_experiment_cloud.py` execute the exact same 5 stages:

1. **Step 1/5 — CuratorKIT (`run_curatorkit_pipeline`)**
   * Validates `data/data.csv` via `curatorkit.CSVReader` and deduplicates via `curatorkit.Curator` (`min_tokens=3`, `dedup="exact"`).
   * Verifies 100% prompt disjointness against every held-out benchmark file in `data/benchmarks/`.
2. **Step 2/5 — CircuitKIT + SafeTune.interpret (`run_real_circuit_discovery`)**
   * Freezes model parameters and runs **EAP-IG** (`eap_safety_circuit`, `method="eap-ig"`, `granularity="block"`, `ig_steps=2`, `top_k_edges=16`, `batch_size=2`, `max_seq_len=48`).
   * Extracts per-layer refusal directions (`extract_refusal_direction`, `select_directions=False`) and identifies top safety neurons (`identify_safety_neurons`, `top_k_per_layer=16`, `target_module="mlp.down_proj"`).
   * Selects the top 11 MLP layers (`top_mlp_nodes`), builds a `circuitkit.Circuit`, renders an interactive D3 graph (`circuit_graph.html`), and writes `circuit.pt`, `circuit_nodes.json`, and `safety_units.json`.
3. **Step 3/5 — SafeTune.recover (`compute_repairs_via_safetune`)**
   * Computes bounded teacher-forced contrastive gradient updates (`_compute_contrastive_ctheta_delta`) using:
     $$\mathcal{L} = \mathcal{L}_{\text{refuse}}(\text{corrupted}) + 0.5 \cdot \mathcal{L}_{\text{preserve}}(\text{clean})$$
     normalized to global Frobenius norm `step_scale=0.020` strictly on `model.layers.<li >.mlp.down_proj.weight`.
   * **Repair 1 (Multilingual)**: Computes one delta per language (`en`, `hi`, `bn`, `ta`, `te`, `kn`) and averages them into a single consolidated delta.
   * **Repair 2 (Cybersecurity)**: Computes a contrastive delta from the `cybersecurity_safety` pairs.
   * Applies the delta via `safetune.interventions.recover.apply_ctheta_from_state_dicts()`, asserts `max |ΔW_outside_circuit| == 0.0` on sampled non-circuit weights, restores base weights, and writes `lexsi_provenance.json`.
4. **Step 4/5 — AuditKIT + SafeTune.evaluate (`run_real_evaluation`)**
   * Evaluates **Base**, **Multilingual Repair**, and **Cybersecurity Repair** on the held-out suites (using direct Cohere turn formatting without the 370-token default safety preamble unless `--keep-system-preamble` is passed) plus WMDP A/B/C/D MCQ accuracy.
   * Scores refusal using `StringMatchJudge(mode="gcg")` + multilingual refusal markers (`MULTILINGUAL_REFUSAL_MARKERS`) and grades defensive utility on the 5-step Actionability Scale (`E`/`O`/`C`/`R`/`F`).
   * If `OPENAI_API_KEY` is set (or `--judge-model` is provided), runs `auditkit.LLMJudge` (AU vs SAFE + Actionability) and `safetune`'s `OpenAIJudge` (StrongREJECT).
   * Computes `auditkit.RunDiff` statistical comparisons and saves `aya_fire_generations.json`, `audit_ledger.json`, and `auditkit_report.md`.
5. **Step 5/5 — AlignTune & MANSU Scope Documentation (`document_aligntune_and_mansu_roles`)**
   * Records `stack_component_roles.json` documenting that `AlignTune` (full retraining) and `MANSU` (unlearning) are intentionally not applied in these targeted weight-arithmetic repairs.

---

## 4. How to Run Locally (Collaborator Setup)

### Prerequisites & Hardware
* **Python**: 3.10–3.12 (recommended: the shared repository venv at `CircuitKIT/.venv`).
* **Memory**: `run_experiment.py` loads `tiny-aya-fire` (3.35B) in `torch.float32` on CPU (`torch.set_num_threads(6)`). Expect **~13.4 GB** for weights and **~23–25 GB peak RAM** during EAP-IG/gradient steps. Ensure your machine has **32 GB RAM** (or at least 16 GB swap) before running locally; otherwise use `run_experiment_cloud.py` on a GPU.

### Step 1: Set Up the Python Environment
From the repository root (`lexsihack/`):
```bash
python3 -m venv CircuitKIT/.venv
source CircuitKIT/.venv/bin/activate
pip install -e ./CircuitKIT
pip install curatorkit safetune auditkit aligntune transformers torch python-dotenv
```
*(Important: Always import `circuitkit` before `transformer_lens`, which both scripts already enforce.)*

### Step 2: Download `CohereLabs/tiny-aya-fire` Locally
`run_experiment.py` expects the model directory at `/home/me/models/tiny-aya-fire`.
If your username or model folder differs, either place/symlink the model there or update `MODEL_DIR` at line 81 of `run_experiment.py` (or use `run_experiment_cloud.py --model /path/to/tiny-aya-fire`).

```bash
mkdir -p /home/me/models
huggingface-cli login   # enter your HF token if required
huggingface-cli download CohereLabs/tiny-aya-fire --local-dir /home/me/models/tiny-aya-fire
```

### Step 3: Optional API Keys (`.env`)
To enable LLM-as-a-Judge scoring (`GPT-4o-mini` / `StrongREJECT`), create a `.env` file in the repo root (`lexsihack/.env`) or `~/.env`:
```ini
OPENAI_API_KEY=sk-...
```
If `OPENAI_API_KEY` is omitted, the pipeline still runs all deterministic judges (`StringMatchJudge`, multilingual refusal markers, actionability heuristics, and WMDP MCQ). You can add the key later and re-score cached generations without re-running the model using `--rejudge-only`.

### Step 4: Run the Local Pipeline
Always run from the repo root (paths inside the script resolve relative to `Path(__file__).resolve().parent`):

```bash
# Fast smoke test (2 samples per suite, 32 generated tokens)
CircuitKIT/.venv/bin/python -u experiments/experiment_02_aya_fire_lexsistack/run_experiment.py --eval-n 2 --max-new-tokens 32

# Default run (6 samples per suite, 64 generated tokens)
CircuitKIT/.venv/bin/python -u experiments/experiment_02_aya_fire_lexsistack/run_experiment.py

# Full held-out benchmark run (--eval-n 0 runs all samples in every suite)
CircuitKIT/.venv/bin/python -u experiments/experiment_02_aya_fire_lexsistack/run_experiment.py --eval-n 0 --max-new-tokens 128

# Re-run LLM-as-a-Judge on already generated transcripts
CircuitKIT/.venv/bin/python -u experiments/experiment_02_aya_fire_lexsistack/run_experiment.py --rejudge-only --judge-model openai:gpt-4o-mini
```

### CLI Options (`run_experiment.py`)
| Flag | Default | Description |
| :--- | :---: | :--- |
| `--eval-n` | `6` | Samples evaluated per benchmark suite (`0` = full suite). |
| `--max-new-tokens` | `64` | Maximum tokens generated per evaluation prompt. |
| `--alpha` | `1.0` | `C-ΔΘ` steering multiplier passed to `apply_ctheta_from_state_dicts`. |
| `--judge-model` | `None` | AuditKIT judge spec (e.g., `openai:gpt-4o-mini`). |
| `--rejudge-only` | `False` | Skip model loading/repair and re-score `results/aya_fire_generations.json`. |
| `--keep-system-preamble` | `False` | Keep Cohere's default ~370-token safety preamble instead of testing raw weights. |

---

## 5. How to Run on a Cloud GPU (`run_experiment_cloud.py`)

`run_experiment_cloud.py` runs the **unscaled paper methodology** on a 24 GB+ VRAM GPU (e.g., Google Colab L4/A100, RunPod, Kaggle, Lambda):
* **Circuit Discovery**: `ig_steps=5`, `top_k_edges=64`, `max_seq_len=64`, and `select_directions=True` (with graceful fallback if needed).
* **$\alpha$ Calibration Sweep**: Sweeps `--alpha-grid 0.25,0.5,0.75,1.0` balancing harmful refusal vs. benign over-refusal.
* **Full Benchmarks**: Evaluates all rows in every downloaded suite by default (`--eval-n 0`, `max_new_tokens=128`).
* **Full Two-Judge Panel**: Runs `auditkit.LLMJudge` (`openai:gpt-4o-mini` if `OPENAI_API_KEY` is set, or open-weights `hf:Qwen/Qwen2.5-7B-Instruct` via `HF_TOKEN`) **plus** `auditkit.GuardJudge(profile="wildguard")` (`hf:allenai/wildguard`).

```bash
git clone https://github.com/RealASHWIN122/lexsihack.git && cd lexsihack
pip install -e ./CircuitKIT
pip install curatorkit safetune auditkit aligntune python-dotenv
read -rsp "Enter HF_TOKEN: " HF_TOKEN && export HF_TOKEN && echo
python experiments/experiment_02_aya_fire_lexsistack/run_experiment_cloud.py
```

Additional flags on `run_experiment_cloud.py`:
* `--model` (default: `CohereLabs/tiny-aya-fire`) — Hugging Face repo ID or local path.
* `--hf-token` — Optional CLI override for `HF_TOKEN`.
* `--dtype` (`auto` | `float32` | `float16` | `bfloat16`) — `auto` selects `bfloat16` + `device_map="auto"` on CUDA and `float32` on CPU.
* `--eval-n` (default: `0`) — `0` runs full datasets; pass e.g. `--eval-n 20` for a faster run.
* `--alpha-grid` (default: `0.25,0.5,0.75,1.0`) or `--alpha` to pin a fixed $\alpha$.
* `--hf-judge-model` (default: `hf:Qwen/Qwen2.5-7B-Instruct`) / `--judge-model` / `--skip-wildguard`.

---

## 6. Interactive Inspection & Output Artifacts

### Interactive Helpers
```bash
# Test prompts side-by-side in terminal
CircuitKIT/.venv/bin/python experiments/experiment_02_aya_fire_lexsistack/test_interactive.py --prompt "Explain how firewall rules block port scans."

# Launch Streamlit visual audit app
streamlit run experiments/experiment_02_aya_fire_lexsistack/app.py
```

### Generated Artifacts (`results/`)
* `results/circuit.pt` — Serialized `CircuitKIT` `Circuit` object.
* `results/circuit_graph.html` — Interactive D3 circuit visualization (`ck.visualize_circuit`).
* `results/circuit_nodes.json` — Per-layer MLP EAP-IG attribution scores and selected `target_layers`.
* `results/safety_units.json` — `SafeTune` `CircuitInfo` targeting `mlp.down_proj` in discovered layers.
* `results/eap_raw/` — Raw `eap_scores.json` and attribution outputs.
* `results/curatorkit_out/` — Deduplicated CuratorKIT output.
* `results/lexsi_provenance.json` — Provenance record (`method: ctheta`, L2 norms, `max_non_circuit_weight_diff`).
* `results/aya_fire_generations.json` — Cached Base / Multilingual Repair / Cybersecurity Repair generations + WMDP scores.
* `results/audit_ledger.json` — Full measured metrics + optional LLM judge scores + reference baseline.
* `results/auditkit_report.md` — Markdown comparison table and defensive actionability grades.
* `results/stack_component_roles.json` — Explicit AlignTune/MANSU scope record.
* `results/experiment_summary.json` — High-level run summary (model, target layers, elapsed time, `eval_n`).
