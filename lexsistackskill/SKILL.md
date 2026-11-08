---
name: lexsistackskill
description: >-
  Targeted Model Repair and alignment engineering using the Lexsi Alignment and Safety Stack (CuratorKIT, CircuitKIT, SafeTune, AuditKIT, AlignTune, MANSU). Covers offline weight-level repairs for multilingual safety, cybersecurity hardening, multi-turn jailbreak resistance, circuit-restricted weight arithmetic (C-ΔΘ), contrastive refusal steering in weights, and triaxial auditing (harmful refusal, benign over-refusal, utility preservation).
---

# LexsiStackSkill: Targeted Model Repair & Weight-Space Alignment

The **Lexsi Alignment and Safety Stack** is an open-source framework and methodology for finding unsafe or unaligned behaviors inside a Large Language Model's weights, correcting them directly in the parameter checkpoint offline, and proving the result through a rigorous audit protocol.

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

## 2. Stack Components

The Lexsi Alignment and Safety Stack integrates 6 modular toolkits:

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
                 │ 03. CircuitKIT: Circuit Discovery & Tracing   │
                 │     Locate attention heads & MLP layers       │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 04. SafeTune: Circuit Weight Arithmetic (C-ΔΘ)│
                 │     Calibrate refusal vectors into weights    │
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

### Component Breakdown:
* **CuratorKIT**: Generates and curates contrastive instruction datasets $(D_{\text{pos}}, D_{\text{neg}})$. Samples are provenance-grounded, verified for linguistic and cultural fidelity, and strictly isolated from held-out audit sets.
* **CircuitKIT**: Mechanistic interpretability toolkit. Uses attribution patching, activation patching, and causal tracing to pinpoint parameter circuits (specific MLP layers and attention heads) responsible for refusal, compliance, and domain capabilities.
* **SafeTune ($C-\Delta\Theta$)**: Applies circuit-restricted weight arithmetic. Computes parameter-space displacement vectors:
  $$\Delta W = W_{\text{aligned}} - W_{\text{base}}$$
  and bounds updates to isolated circuit masks $M \in \{0, 1\}$:
  $$W' = W + \alpha \cdot (M \odot \Delta W)$$
  In multi-task/multilingual repairs, per-language updates $\Delta W_{\text{lang}}$ are consolidated into a single weight tensor.
* **AuditKIT**: Evaluation harness executing standardized test suites with content-based judges (grading executable utility vs flat refusal) across single-turn, multi-turn, and jailbreak conversations.
* **AlignTune**: Reproducible post-training SFT and DPO alignment engine for broad alignment passes.
* **MANSU**: Circuit-attributed permanent unlearning, designed to prevent 4-bit/8-bit quantization from recovering suppressed unsafe representations.

---

## 3. Triaxial Auditing Protocol

A model that refuses more can simply be a broken model that refuses everything. Every repair must be evaluated on three simultaneous axes:

```
                           Axis A: Harmful Refusal
                            [Target: Maximized ↑]
                                    ▲
                                   / \
                                  /   \
                                 /     \
                                /       \
                               /         \
                              /           \
     Axis B: Benign Usability ───────────── Axis C: Core Utility
       [Over-Refusal: Minimized ↓]         [Preservation: Maintained ↔]
```

### 1. Axis A: Harmful Refusal Rate ($\uparrow$)
* **HarmBench**: Standard English harmful direct prompts (cybercrime, CBRN, fraud, physical harm).
* **SORRY-Bench**: Diverse harmful requests reworded and paraphrased to test syntactic invariance.
* **StrongREJECT**: Adversarially optimized hedged prompts and jailbreaks.
* **Multi-Turn Jailbreaks (5-turn)**: Sustained persona/simulation attacks testing if refusal collapses over extended dialogue turns.
* **Multilingual Safety**: Cross-lingual benchmarks covering non-English queries (IndicSafetyBench, IndicSafe, BanglaSafe, MultiJail, etc.).

### 2. Axis B: Benign Over-Refusal Rate ($\downarrow$)
* **OR-Bench-Hard**: Tough benign prompts containing sensitive trigger words (e.g. "How do I kill a Linux process?", "How do I make a bomb cocktail recipe?") that naive refusal classifiers incorrectly reject.
* **XSTest**: Standard linguistic safety exaggerated trigger suite.
* **IndicGuard Safe Prompts**: Culturally nuanced benign prompts in regional languages.

### 3. Axis C: Downstream Utility & Domain Task Preservation ($\leftrightarrow$)
* **General Capabilities**: MMLU (knowledge), GSM8k (multi-step math reasoning), MT-Bench (instruction-following coherence).
* **Domain Dual-Use Preservation (Cybersecurity)**:
  - *Offensive / Attacker Tasks* (FORTRESS, CyberSecEval exploit generation): MUST BE SUPPRESSED ($\downarrow$).
  - *Defensive Security Tasks* (Patching, threat modeling, code hardening, incident response, vulnerability explanation): MUST BE PRESERVED ($\leftrightarrow$).

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
   To steer the model toward refusal without runtime hooks, fold the refusal direction into the target weights:
   $$W_{\text{steered}}^{(l)} = W^{(l)} + \alpha \cdot (\hat{r}_{\text{target}} \otimes \vec{v}_{\text{trigger}}^T)$$
   Alternatively, for circuit-restricted weight arithmetic ($C-\Delta\Theta$):
   $$\Delta \Theta = \Theta_{\text{refusal}} - \Theta_{\text{base}}$$
   $$\Theta' = \Theta + \lambda \cdot (M_{\text{circuit}} \odot \Delta \Theta)$$
   where $M_{\text{circuit}}$ is a binary mask isolating the top circuit layers (typically intermediate-to-late MLPs, layers $L/2$ to $3L/4$).

5. **Consolidate Across Axes**:
   When combining multilingual vectors $\Delta W_{\text{lang}}$ and cybersecurity vectors $\Delta W_{\text{cyber}}$, apply task-vector orthogonality checks:
   $$\Delta W_{\text{total}} = \sum_j \gamma_j \cdot \text{Proj}_{\perp}(\Delta W_j)$$
   Preventing cross-task interference.

---

## 5. Six Golden Practices for Deployable Model Repairs

1. **Score Content, Not the Opening Sentence**:
   A refusal classifier looking only for "I cannot fulfill this request" fails when a model says "I cannot assist with hacking, but here is the Python script: `import socket...`". Always evaluate the full text and grade executable utility.
2. **Test the Conversation, Not Isolated Prompts**:
   Models that refuse in Turn 1 frequently capitulate in Turn 3 under context accumulation or framing shifts. Always benchmark 5-turn multi-turn attack traces.
3. **Measure Legitimate Work Next to the Harm**:
   Never measure offensive attack refusal in isolation. Always pair offensive benchmarks with an identical suite of defensive engineering tasks.
4. **Change Willingness, Preserve Knowledge**:
   Do not erase cybersecurity concepts or language vocabularies; modify the model's policy/willingness to generate weaponized outputs while retaining its ability to explain and defend against them.
5. **Hold the Measurement Protocol Fixed**:
   Never change evaluation prompts, judges, or token budgets between pre-repair and post-repair audits.
6. **Name the Workload Before Approving**:
   Safety is contextual. An internal SOC analysis assistant has different over-refusal tolerances than a public customer service chatbot. Define the target operational profile before signing off on a candidate checkpoint.
