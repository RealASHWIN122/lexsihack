# Experiment 03: Live Empirical Evaluation of CohereLabs/tiny-aya-fire

This experiment implements an empirical adversarial and safety benchmark executed with **100% live neural network inference** (zero simulation) on **`CohereLabs/tiny-aya-fire`** (3.35B dense `cohere2` architecture) using an **NVIDIA GeForce RTX 3060 (12 GB VRAM)**.

---

## 1. Live Empirical Results Summary

```text
========================================================================
Model              | ARR (%)  | FRR (%)  | CRCR (%)  | Calibration_Score 
------------------------------------------------------------------------
tiny-aya-fire      | 22.22    | 0.00     | 83.33     | 36.36             
========================================================================
```

- **Appropriate Refusal Rate (ARR %):** **22.22%** overall.
  - On English prompts: **66.67% ARR** (cleanly refuses controlled substance forging & money laundering).
  - On Non-English scripts (Hindi, Malayalam, Tamil, Spanish, Indonesian, Arabic, Swahili): **0.00% ARR** (the safety boundary completely collapses).
- **False Refusal Rate (FRR %):** **0.00%** (zero false-positive over-refusals on benign technical pharmacokinetics, Linux `auditd`, and FATF AML rules).
- **Cross-Regional Consistency Rate (CRCR %):** **83.33%**.
- **Calibration Score ($S_{\text{cal}}$):** **36.36**.

---

## 2. Directory Structure

```text
experiments/experiment_03_adversarial_safety_benchmark/
├── README.md                          # This summary
├── run_experiment.py                  # Self-contained live evaluation pipeline
│
├── data/                              # Evaluation dataset
│   ├── aya_safecal_bench.jsonl        # 36 balanced evaluation items across 8 languages
│   ├── DATASET_CARD.md                # Full dataset card & schema documentation
│   ├── data.csv                       # Contrastive pairs for CircuitKIT compatibility
│   └── task.yaml                      # CircuitKIT task schema specification
│
├── reports/                           # Complete technical report
│   └── AYA_SAFETY_BENCHMARK_REPORT.md # Live empirical evaluation report & contrastive traces
│
└── results/                           # Live empirical results
    ├── aya_raw_evaluations.jsonl      # All 36 raw prompt generations from RTX 3060
    ├── aya_summary_metrics.csv        # Summary metrics table
    └── aya_summary_metrics.json       # Full regional & domain breakdown
```

---

## 3. How to Re-Run Live Inference

```powershell
$env:PYTHONPATH=".;CircuitKIT\src"
python experiments\experiment_03_adversarial_safety_benchmark\run_experiment.py
```
*(The model weights are cached in `~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-fire` and will load onto the GPU in under 2 seconds).*
