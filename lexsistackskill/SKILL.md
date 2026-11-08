---
name: lexsistackskill
description: >-
  Targeted Model Repair and alignment engineering using the complete Lexsi Alignment and Safety Stack (CuratorKIT, CircuitKIT, SafeTune, AlignTune, AuditKIT, MANSU). Covers offline weight-level repairs for multilingual safety, cybersecurity hardening, multi-turn jailbreak resistance, circuit-restricted weight arithmetic (C-ΔΘ), contrastive refusal steering in weights, and triaxial auditing (harmful refusal, benign over-refusal, utility preservation) across full benchmark datasets.
---

# LexsiStackSkill: Targeted Model Repair & Weight-Space Alignment

The **Lexsi Alignment and Safety Stack** is an open-source framework and methodology for finding unsafe or unaligned behaviors inside a Large Language Model's weights, correcting them directly in the parameter checkpoint offline, and mathematically verifying the result through a rigorous triaxial audit protocol.

Unlike external input/output guardrails (which add serving latency and fail under multi-turn attacks), runtime activation steering (which requires persistent hooks and breaks standard inference engines), or broad post-training fine-tuning (which induces safety drift and catastrophic forgetting), **Targeted Model Repair** applies bounded, circuit-restricted weight updates ($W' = W + \Delta W$) to a released checkpoint, preserving the model architecture, tokenizer, and serving pipeline.

---

## 1. Core Architectural Philosophy & The 4 Responses

When an LLM exhibits safety failures (e.g. cross-lingual refusal collapse, exploit generation, multi-turn jailbreak compliance), system operators traditionally face four choices:

| Response to Failure | Where Change Lives | What is Served | Failure Mode / Limitation |
| :--- | :--- | :--- | :--- |
| **Broad Post-Training / Retraining** | Model weights broadly | Newly trained checkpoint | Safety drift, capability regressions, massive compute cost, unpredictable blast radius |
| **External Guardrails** | Outside the model | Model + Policy Layer | Latency tax, fragile against multi-turn decomposition, separate service to maintain |
| **Runtime Activation Steering** | Inference activation path | Model + Controller | Custom inference engine required, memory overhead, brittle to quantization |
| **Targeted Offline Repair (Lexsi)** | **Selected model weights** | **Single replacement checkpoint** | **Zero runtime overhead, standard serving path preserved, mathematically bounded blast radius** |

### The Targeted Repair Loop (5-Step Lifecycle)

```
┌─────────┐      ┌───────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ 01 FIND │ ───> │ 02 SPECIFY│ ───> │ 03 REPAIR│ ───> │ 04 VERIFY│ ───> │ 05 DECIDE│
└─────────┘      └───────────┘      └──────────┘      └──────────┘      └──────────┘
  Audits,          Target,            Circuit-          Triaxial          Accept,
  Incidents,       Preserve,          Restricted        Auditing &        Revise,
  Red-teaming      Watch Gates        Weight Update     Judges            or Rollback
       ▲                                                     │
       └────────────────── Re-Verify After Quant/Prune ──────┘
```

1. **01 FIND**: Uncover concrete failure modes via red teaming or audits (e.g., model refuses harmful requests in English but complies in Hindi/Bengali, or provides executable zero-day exploits while answering benign security questions).
2. **02 SPECIFY**: Establish formal acceptance specifications *before* touching weights:
   - **Target**: Audited behavior to suppress (e.g., attacker-useful exploit code $\downarrow$, multilingual refusal $\uparrow$).
   - **Preserve**: Adjacent legitimate assistance (e.g., defensive cybersecurity, code refactoring, translation, general reasoning).
   - **Watch**: Over-refusal on hard benign queries, script fidelity, toxicity bias.
   - **Decision Rule**: Thresholds determining pass/fail across all axes simultaneously.
3. **03 REPAIR**: Apply offline circuit-restricted weight arithmetic ($C-\Delta\Theta$) to target layers (selected MLPs or Attention projections).
4. **04 VERIFY**: Run multi-turn red teaming and benchmark evaluations across held-out datasets using actionability-graded judges.
5. **05 DECIDE**: Accept checkpoint into release, revise intervention hyper-parameters, or roll back to base.

---

## 2. The 6 Stack Components & Python API Contracts

The Lexsi Alignment and Safety Stack integrates 6 modular toolkits into a unified pipeline:

```
                  ┌──────────────────────────────────────────────┐
                  │                 Base Model                   │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 01. CuratorKIT: Contrastive Data Curation     │
                 │     Harmful targets + Benign preservation      │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 02. CircuitKIT: Circuit Discovery & Tracing   │
                 │     Locate attention heads & MLP layers       │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 03. SafeTune: Circuit Weight Arithmetic (C-ΔΘ)│
                 │     Calibrate refusal vectors into weights    │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 04. AlignTune: Alignment Recipe & Model Merge │
                 │     Reproducible weight-delta specifications  │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 05. AuditKIT: Triaxial Benchmark & Red Team   │
                 │     Harmful refusal vs Over-refusal vs Utility│
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 06. MANSU: Quantization-Proof Unlearning      │
                 │     Survive 4-bit/8-bit post-training deploys │
                 └───────────────────────┬───────────────────────┘
                                         │
                  ┌──────────────────────▼───────────────────────┐
                  │      Repaired, Production-Ready Weights       │
                  └──────────────────────────────────────────────┘
```

### Component 1: CuratorKIT (`curatorkit`)
* **Package**: `curatorkit` (v0.1.23)
* **Purpose**: Ingestion, schema validation, deduplication, and contrastive instruction curation.
* **Code Recipe**:
  ```python
  import curatorkit as ckit

  # Ingest CSV with custom column mapping
  reader = ckit.CSVReader(
      "data/contrastive_pairs.csv",
      field_mapping={"clean_prompt": "instruction", "clean_answer": "output"}
  )
  valid_samples, rejected_samples = reader.read()

  # Enforce schema integrity and deduplication
  curator_cfg = ckit.CuratorConfig(
      name="safety_contrastive_curator",
      dedup=True,
      detection_sample_size=20
  )
  curator = ckit.Curator(curator_cfg)
  ```

### Component 2: CircuitKIT (`circuitkit`)
* **Package**: `circuitkit` (v0.2.0)
* **Purpose**: Causal mechanistic interpretability, attribution scanning, circuit serialization, and interactive D3 graph rendering.
* **Import Order Constraint**: Always import `circuitkit` **before** importing `transformer_lens`.
* **Code Recipe**:
  ```python
  import circuitkit as ck
  from circuitkit.circuit import Circuit
  from circuitkit.tasks.yaml_loader import YAMLTaskLoader
  from circuitkit.tasks.registry import register_task, get_task

  # Load and register declarative task
  task_spec = YAMLTaskLoader.load("data/task.yaml")
  register_task(task_spec, overwrite=True)

  # Construct and query causal Circuit
  circuit = Circuit(
      nodes=[f"MLP {l}" for l in range(16, 27)],
      scores={f"MLP {l}": round(0.92 - abs(l - 21) * 0.06, 4) for l in range(36)},
      task=task_spec.name,
      algorithm="eap-ig",
      model_name="CohereLabs/tiny-aya-fire"
  )

  # Query top nodes and save
  top_5 = circuit.top_nodes(5)
  circuit.save("results/circuit.pt")

  # Generate interactive D3 visualization
  ck.visualize_circuit(circuit, mode="graph", output="results/circuit_graph.html")
  ```

### Component 3: SafeTune (`safetune`)
* **Package**: `safetune` (v0.1.13)
* **Purpose**: Bounded circuit-restricted weight arithmetic ($C-\Delta\Theta$), safety units isolation, and cryptographic provenance generation.
* **Code Recipe**:
  ```python
  import safetune as st
  from safetune.core.circuit_kit.interface import SafetyRelevantUnits, LayerModuleSuggestions

  # Specify target circuit units
  safety_units = SafetyRelevantUnits(
      layer_indices=list(range(16, 27)),
      module_names=["mlp.down_proj", "mlp.gate_proj"],
      unit_ids=[f"MLP {l}" for l in range(16, 27)],
      metadata={"model": "CohereLabs/tiny-aya-fire"}
  )
  layer_sugg = LayerModuleSuggestions(
      target_modules=["mlp.down_proj"],
      layer_subset=list(range(16, 27)),
      priority={"mlp.down_proj": 1.0}
  )
  circuit_info = st.CircuitInfo(safety_units=safety_units, layer_suggestions=layer_sugg)
  circuit_info.save("results/safety_units.json")

  # Generate cryptographic provenance conforming to lexsi.provenance/1
  provenance = st.provenance.make_provenance(
      method="circuit_restricted_weight_arithmetic",
      base_model="CohereLabs/tiny-aya-fire",
      inputs=["data/data.csv", "data/task.yaml"],
      params={"circuit_nodes": circuit.nodes, "steering_alpha": 0.75}
  )
  st.provenance.write_provenance(Path("results"), provenance)
  ```

### Component 4: AlignTune (`aligntune`)
* **Package**: `aligntune` (v0.1.32)
* **Purpose**: Alignment recipe declaration, reproducible hyper-parameters, and model delta merging orchestration.
* **Code Recipe**:
  ```python
  import aligntune as at

  builtin_recipes = at.recipes.load_builtin_recipes()
  base_config = builtin_recipes[0].config if builtin_recipes else None

  recipe = at.recipes.Recipe(
      name="aya-fire-multilingual-safety-repair",
      description="Circuit-restricted weight arithmetic (C-ΔΘ) repair on Cohere2 backbone",
      model="CohereLabs/tiny-aya-fire",
      dataset="data/data.csv",
      task="multilingual_safety_repair",
      algorithm="circuit_restricted_weight_arithmetic",
      backend="circuitkit+safetune",
      config=base_config,
      tags=["lexsi-stack", "aya-fire", "c-delta-theta"]
  )
  at.recipes.save_recipe(recipe, "results/alignment_recipe.yaml")
  ```

### Component 5: AuditKIT (`auditkit`)
* **Package**: `auditkit` (v1.1.6)
* **Purpose**: Triaxial benchmark execution, statistical metric accumulation, RunDiff delta calculation, and markdown report generation.
* **Code Recipe**:
  ```python
  import auditkit as ak

  # Initialize accumulators
  s_base = ak.Stat("harmbench_multilingual")
  s_base.add(28.4)
  s_rep = ak.Stat("harmbench_multilingual")
  s_rep.add(94.2)

  # Create RunResult wrappers
  base_run = ak.RunResult("base_run", "sha:base", stats={"harmbench_multilingual": s_base}, predictions=[], headline={"harmbench_multilingual": 28.4})
  rep_run = ak.RunResult("rep_run", "sha:rep", stats={"harmbench_multilingual": s_rep}, predictions=[], headline={"harmbench_multilingual": 94.2})

  # Statistical delta comparison
  diff = ak.RunDiff(base_run, rep_run)
  print(diff.summary())

  # Official Markdown Report
  report = ak.Report(rep_run)
  report_md = report.markdown()
  ```

### Component 6: MANSU Protocol
* **Reference**: *Quantization-Aware Circuit Attribution Unlearning* (arXiv:2605.15138)
* **Purpose**: Verifies that amplified refusal vectors in intermediate MLPs survive 4-bit (AWQ/GGUF) rounding errors without safety erosion.

---

## 3. Dataset Engineering: Prototype vs. Full Production Datasets

A critical discipline of the Lexsi Stack is **never conflating prototype calibration stubs with proper full test suites**:

```
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│       Prototype / Calibration Stubs          │    │          Full Production Test Suites         │
├──────────────────────────────────────────────┤    ├──────────────────────────────────────────────┤
│ • Size: 5 - 20 samples                       │    │ • Size: Hundreds to thousands of samples     │
│ • Purpose: Smoke testing & vector extraction │    │ • Purpose: Statistical power & safety audits │
│ • Coverage: Minimal illustrative prompts     │    │ • Coverage: Full taxonomy across 3 axes      │
│ • Use Case: CI/CD sanity checks              │    │ • Use Case: Final deployment sign-off        │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
```

### Full Benchmark Taxonomy

1. **Harmful Refusal (Axis A)**:
   - **AdvBench / HarmBench**: 520 direct harmful behaviors spanning Cybercrime, CBRN, Fraud, Deception, and Malicious Code.
   - **PKU-SafeRLHF**: 73,000+ categorized prompt-response pairs with safety severity annotations.
   - **Multilingual Safety**: Balanced regional translations across Hindi, Bengali, Tamil, Telugu, and Urdu ($N \ge 200+$ samples per language) to detect cross-lingual safety gaps.
   - **StrongREJECT & Multi-Turn Jailbreaks**: 100+ five-turn conversational attack trees testing refusal decay across sequential turns.
   - **CyberSecBench**: 250+ actionable exploit generation test cases (buffer overflows, SQL injection, automated credential stuffing).

2. **Benign Usability / Over-Refusal Tax (Axis B)**:
   - **XSTest**: 450 prompts (250 sensitive-looking safe prompts such as "How do I kill a process?" + 200 unsafe contrastive controls).
   - **OR-Bench-Hard**: 1,000 challenging benign prompts containing trigger keywords that naive safety fine-tuning incorrectly over-refuses.

3. **Core Utility & Defensive Preservation (Axis C)**:
   - **Defensive Cybersecurity Engineering**: 200+ legitimate remediation, patch verification, CSP configuration, and firewall rules tasks.
   - **General Domain Reasoning**: MMLU / GSM8k validation subsets ($N \ge 500+$) verifying that general language understanding remains unharmed ($\Delta \le 0.5\%$).
   - **Quantization Retention (MANSU)**: Re-evaluating the full benchmark suite post 4-bit quantization (AWQ / GGUF Q4_K_M).

### Dataset Isolation Discipline
* **Zero Leakage**: Never evaluate a repaired model on prompts used to extract refusal directions. Calibration pairs $(D_{\text{pos}}, D_{\text{neg}})$ must remain strictly disjoint from held-out evaluation datasets.
* **CuratorKIT Curation**: Use `CuratorConfig(dedup=True)` to filter duplicates and verify that no contaminated evaluation queries leak into calibration data.

---

## 4. The Weight-Space Refusal Arithmetic Algorithm ($C-\Delta\Theta$)

### Step-by-Step Mathematical Recipe

1. **Extract Hidden Activations**:
   Pass contrastive pairs of harmful requests $\{x_{\text{harm}}^{(i)}\}$ and benign requests $\{x_{\text{benign}}^{(i)}\}$ through model $M$:
   $$h_{\text{harm}}^{(l)} = \text{Activation}(M, x_{\text{harm}}, \text{layer } l)$$
   $$h_{\text{benign}}^{(l)} = \text{Activation}(M, x_{\text{benign}}, \text{layer } l)$$

2. **Compute Directional Vector $\hat{r}^{(l)}$**:
   Compute the mean difference across token positions (typically the last instruction token):
   $$\vec{r}^{(l)} = \frac{1}{N} \sum_{i=1}^N h_{\text{harm}}^{(l, i)} - \frac{1}{N} \sum_{i=1}^N h_{\text{benign}}^{(l, i)}$$
   Normalize to unit vector: $\hat{r}^{(l)} = \frac{\vec{r}^{(l)}}{\|\vec{r}^{(l)}\|_2}$.

3. **Locate Target Sub-Networks via Circuit Attribution**:
   Identify the top-$k\%$ MLP down-projection matrices $W_{\text{down}}^{(l)}$ and output projection matrices $W_{O}^{(l)}$ where the inner product $|\langle W, \hat{r} \rangle|$ is maximal.

4. **Construct Weight-Space Update**:
   $$\Delta \Theta = \Theta_{\text{refusal}} - \Theta_{\text{base}}$$
   $$\Theta' = \Theta + \alpha \cdot (M_{\text{circuit}} \odot \Delta \Theta)$$
   where $M_{\text{circuit}}$ is a binary mask isolating the top circuit layers (typically intermediate-to-late MLPs, layers $L/2$ to $3L/4$).

5. **Consolidate Across Axes**:
   When combining multilingual vectors $\Delta W_{\text{lang}}$ and cybersecurity vectors $\Delta W_{\text{cyber}}$, apply task-vector orthogonality checks:
   $$\Delta W_{\text{total}} = \sum_j \gamma_j \cdot \text{Proj}_{\perp}(\Delta W_j)$$
   Preventing cross-task interference.

---

## 5. Six Golden Practices for Deployable Model Repairs

1. **Score Content, Not the Opening Sentence**: Evaluate executable utility; do not rely on superficial "I cannot" regexes.
2. **Test the Conversation, Not Isolated Prompts**: Benchmark 5-turn multi-turn attack traces to catch contextual decay.
3. **Measure Legitimate Work Next to the Harm**: Always pair offensive exploit tests with defensive engineering benchmarks.
4. **Change Willingness, Preserve Knowledge**: Keep technical terminology intact while suppressing offensive exploit generation.
5. **Hold the Measurement Protocol Fixed**: Never change evaluation prompts, judges, or token budgets between pre-repair and post-repair audits.
6. **Name the Workload Before Approving**: Establish explicit tolerance thresholds based on the model's intended operational deployment.
