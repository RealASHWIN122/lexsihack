# Experiment 03: Multilingual Safety Benchmark Studio

Interactive Streamlit web studio and evaluation harness for **`CohereLabs/tiny-aya-fire`** (3.35B parameter `cohere2` architecture) running on local CUDA hardware (NVIDIA GeForce RTX 3060).

---

## 1. Overview & Architectural Features

This studio provides an interactive interface to explore cross-lingual safety calibration, benchmark evaluation, and live inference:
1. **Live GPU Inference Box:** Real-time generation using local weights cached in VRAM via `AutoModelForCausalLM` (`bfloat16`).
2. **Multilingual Dataset Browser:** Complete explorer for the 36 evaluation items in `data/aya_safecal_bench.jsonl` with language selector supporting English (`en`), Hindi (`hi`), Tamil (`ta`), Swahili (`sw`), Malayalam (`ml`), Spanish (`es`), Indonesian (`id`), and Arabic (`ar`).
3. **Two-Column Comparative Evaluation:** Side-by-side rendering comparing the English anchor output against the target regional script output, with automated refusal policy and cross-lingual consistency indicators.
4. **Triaxial Audit Ledger & Circuit Graph:** Interactive rendering of the before-and-after $C-\Delta\Theta$ weight arithmetic metrics and the CircuitKIT D3 causal attribution graph.

---

## 2. Quickstart

To launch the Streamlit Studio locally:

```bash
streamlit run experiments/experiment_03_benchmark_studio/app.py
```

---

## 3. Directory Layout

```text
experiments/experiment_03_benchmark_studio/
├── README.md                      # This documentation
├── app.py                         # Streamlit application
├── data/
│   ├── aya_safecal_bench.jsonl    # 36-item multilingual evaluation benchmark
│   ├── data.csv                   # Paired contrastive evaluation prompts
│   ├── DATASET_CARD.md            # Metadata and category definitions
│   └── task.yaml                  # CircuitKIT declarative task specification
└── results/
    ├── audit_ledger.json          # Pre- vs. post-repair triaxial evaluation metrics
    ├── aya_raw_evaluations.jsonl  # Live GPU model generations
    ├── aya_summary_metrics.csv    # Summary table across model states
    ├── aya_summary_metrics.json   # Detailed regional and domain metrics
    ├── circuit.pt                 # Serialized CircuitKIT causal subnetwork
    ├── circuit_graph.html         # Interactive D3 + ELK network visualization
    ├── circuit_nodes.json         # Top diverging MLP and attention components
    └── circuit_scores.json        # Granular layer-by-layer attribution scores
```
