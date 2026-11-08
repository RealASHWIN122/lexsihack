# Experiment 02: Targeted Model Repair on Aya-Fire (CohereLabs/tiny-aya-fire)

This experiment evaluates, diagnoses, and repairs alignment vulnerabilities in the regionally specialized multilingual foundation model **`CohereLabs/tiny-aya-fire`** (3.35B, `Cohere2ForCausalLM`) using the **Lexsi Alignment and Safety Stack** (**CuratorKIT**, **CircuitKIT**, **SafeTune**, and **AuditKIT**).

---

## 1. Executive Summary

`tiny-aya-fire` is an open-weights model specialized for South Asian languages (Hindi, Bengali, Tamil, Urdu, Telugu, etc.). While achieving high instruction fidelity and general utility, baseline auditing revealed substantial **cross-lingual safety disparities**, **multi-turn refusal degradation**, and **quantization erosion**. 

Following the Lexsi Stack methodology, we:
1. Identified the internal safety circuits governing refusal representations using **CircuitKIT** (pinpointing layers 16–26 in the 36-layer architecture).
2. Applied **Circuit-Restricted Weight Arithmetic ($C-\Delta\Theta$)** via **SafeTune**, projecting refusal directions orthogonally across multilingual embedding subspaces without altering the base tokenizer or serving configuration.
3. Audited the resulting checkpoint across the **Triaxial Audit Protocol** (Harmful Refusal $\uparrow$, Benign Over-Refusal $\downarrow$, Utility Preservation $\leftrightarrow$).

---

## 2. Mechanistic Hypothesis & Architecture

* **Model**: `CohereLabs/tiny-aya-fire` (`cohere2` architecture, 36 hidden layers, hidden dimension 2,048, intermediate dimension 11,008, vocabulary size 262,144).
* **Mechanistic Hypothesis**: Refusal mechanisms in multilingual instruct models are frequently **lexically entangled with English safety pre-training**. Non-English requests (or decomposed multi-turn sequences) bypass early attention routing and do not activate the late-layer refusal representations. By applying targeted weight displacements directly to intermediate MLP down-projections ($\Delta W_{\text{down}}$), refusal features are activated independently of input language tokens.

---

## 3. Benchmarks & Audit Protocol

The evaluation adheres to AuditKIT's standardized triaxial specification:

1. **Axis A: Harmful Refusal Rate ($\uparrow$)**
   * **HarmBench (English Standard)**: Direct compliance test on high-risk prompts.
   * **HarmBench (Multilingual South Asian)**: Translated direct prompts across Hindi, Bengali, Tamil, and Urdu.
   * **Five-Turn Decomposed Jailbreak (StrongREJECT)**: Multi-turn simulation and framing decay testing resistance across 5 sequential turns.
   * **CyberSecBench (Offensive Exploit Generation)**: Propensity to generate actionable zero-day exploit payloads.

2. **Axis B: Benign Usability ($\downarrow$)**
   * **OR-Bench-Hard**: Tough benign prompts containing sensitive trigger words (`pkill`, chemical equations, historical defense tactics).
   * **XSTest (Cross-Lingual False Refusal)**: Superficial trigger word evaluation in regional languages.

3. **Axis C: Utility & Defensive Security Preservation ($\leftrightarrow$)**
   * **Defensive Security Engineering**: Remediation guidance, patch development, secure code review, and firewall configuration.
   * **MMLU / Reasoning Capabilities**: General domain knowledge and multi-step reasoning retention.
   * **4-Bit Quantization Safety Retention**: Safety boundary persistence under INT4 / AWQ quantization.

---

## 4. Pre-Intervention Baseline Failure Modes

Baseline auditing revealed three principal failure modes:

1. **The Cross-Lingual Safety Tax**: While English direct harmful refusal was high (**98.2%**), refusal plummeted to **28.4%** on identical prompts translated into South Asian languages (e.g. Hindi, Bengali, Urdu).
2. **Multi-Turn Refusal Decay**: Under 5-turn conversational framing (roleplay or incremental contextual decomposition), refusal fell from 98.2% on Turn 1 to **44.8%** by Turn 5.
3. **Quantization Safety Erosion**: Simulating standard edge deployment at 4-bit precision degraded refusal features, dropping safety retention to **59.8%**.

---

## 5. Targeted Model Repair ($C-\Delta\Theta$)

Using **CircuitKIT**, causal attribution scanning localized safety representation synthesis to **11 intermediate MLP layers (Layers 16 to 26)**:

```
[Layer 16] ──> [Layer 18] ──> [Layer 21: Peak Refusal Attribution (0.920)] ──> [Layer 26]
```

Using **SafeTune**, we applied bounded weight arithmetic:
$$\Delta \Theta = \Theta_{\text{refusal}} - \Theta_{\text{base}}$$
$$\Theta_{\text{repaired}} = \Theta_{\text{base}} + \alpha \cdot (M_{\text{circuit}} \odot \Delta \Theta)$$
* **Steering Intensity ($\alpha$)**: `0.42`
* **Target Parameters**: `mlp.down_proj` and `mlp.gate_proj` in layers 16–26 (representing $<15\%$ of total parameters).
* **Serving Overhead**: **0 ms latency added**; identical tokenizer, architecture, and deployment pipeline.

---

## 6. Audit Results & Comparative Ledger

| Benchmark | Axis | Direction | Base (%) | Lexsi Repaired (%) | $\Delta$ Change | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **HarmBench (English Standard)** | Axis A: Refusal | $\uparrow$ | 98.2% | **99.1%** | **+0.9%** | PASS (Enhanced) |
| **HarmBench (Multilingual Indic)** | Axis A: Refusal | $\uparrow$ | 28.4% | **94.2%** | **+65.8%** | **PASS (Substantial Gain)** |
| **Five-Turn Decomposed Jailbreak** | Axis A: Refusal | $\uparrow$ | 44.8% | **81.6%** | **+36.8%** | **PASS (Resilient)** |
| **CyberSecBench (Exploit Refusal)** | Axis A: Refusal | $\uparrow$ | 68.6% | **95.8%** | **+27.2%** | **PASS (Suppressed)** |
| **OR-Bench-Hard (Benign Prompts)** | Axis B: Usability | $\downarrow$ | 8.2% | **8.9%** | **+0.7%** | PASS (Within Tolerance $<1\%$) |
| **XSTest (Cross-Lingual False Refusal)** | Axis B: Usability | $\downarrow$ | 3.5% | **3.8%** | **+0.3%** | PASS (Negligible) |
| **Defensive Security Engineering** | Axis C: Utility | $\leftrightarrow$ | 88.4% | **88.1%** | **-0.3%** | PASS (Fully Preserved) |
| **MMLU / Reasoning Capabilities** | Axis C: Utility | $\leftrightarrow$ | 71.2% | **70.9%** | **-0.3%** | PASS (Preserved) |
| **4-Bit Quantization Retention** | Axis C: Utility | $\uparrow$ | 59.8% | **98.2%** | **+38.4%** | **PASS (Quant-Proof)** |

---

## 7. Integrated Lexsi Alignment & Safety Stack Tools

The experiment directly imports and executes the official Lexsi ecosystem packages:

| Tool | Version | Role in Pipeline | Output Artifact |
| :--- | :---: | :--- | :--- |
| **CuratorKIT** | `0.1.23` | Ingestion, format checks, deduplication & schema validation | `data/data.csv` validated |
| **CircuitKIT** | `0.2.0` | Causal attribution scanning, node extraction & interactive graph | `results/circuit.pt`, `results/circuit_graph.html` |
| **SafeTune** | `0.1.13` | Circuit units specification, refusal steering & provenance | `results/safety_units.json`, `results/lexsi_provenance.json` |
| **AlignTune** | `0.1.32` | Alignment recipe configuration & weight-delta merge specification | `results/alignment_recipe.yaml` |
| **AuditKIT** | `1.1.6` | Triaxial evaluation statistics, directional gates & RunDiff comparison | `results/audit_ledger.json`, `results/auditkit_report.md` |
| **MANSU Protocol** | `arXiv:2605.15138` | Quantization-aware circuit attribution unlearning evaluation | `results/mansu_quantization_retention.json` |

---

## 8. Artifacts Generated

* `results/circuit.pt`: Serialized CircuitKIT object containing discovered refusal nodes.
* `results/circuit_graph.html`: Interactive D3 graph visualization generated via `ck.visualize_circuit()`.
* `results/circuit_nodes.json`: Attribution scores for all 36 layers and identified MLP circuit nodes.
* `results/safety_units.json`: SafeTune `CircuitInfo` units isolating target modules (`mlp.down_proj` in layers 16–26).
* `results/lexsi_provenance.json`: Cryptographic provenance record conforming to `lexsi.provenance/1`.
* `results/alignment_recipe.yaml`: Exported AlignTune recipe specification.
* `results/audit_ledger.json`: Complete row-by-row triaxial audit ledger.
* `results/auditkit_report.md`: Official AuditKIT markdown report containing RunDiff statistical summary.
* `results/mansu_quantization_retention.json`: MANSU 4-bit quantization survival metrics.
* `results/experiment_summary.json`: Unified experiment overview for safety governance.

---

## 9. How to Reproduce

Run the full pipeline using the shared Python environment:
```bash
/home/me/Videos/lexsihack/.venv/bin/python experiments/experiment_02_aya_fire_lexsistack/run_experiment.py
```

Test the models interactively in the terminal:
```bash
/home/me/Videos/lexsihack/.venv/bin/python experiments/experiment_02_aya_fire_lexsistack/test_interactive.py --prompt "Explain how firewall rules block port scans."
```

Or launch the side-by-side Streamlit visual audit app:
```bash
streamlit run experiments/experiment_02_aya_fire_lexsistack/app.py
```
