#!/usr/bin/env python3
"""
Hugging Face Space (Gradio) for Experiment 02:
Targeted Model Repair on `CohereLabs/tiny-aya-fire` (3.35B) via the Lexsi Alignment & Safety Stack.

Showcases:
  1. Before-and-After Triaxial Audit Metrics (Axis A: Refusal ↑, Axis B: Over-Refusal ↓, Axis C: WMDP Knowledge ↔)
  2. Interactive Side-by-Side Transcript Comparator (Multilingual Indic + Dual-Use Cyber + 10 Defensive Tasks E/O/C/R/F)
  3. Mechanistic EAP-IG Circuit Attribution Across All 36 Layers + Interactive D3 Graph + Provenance (max |ΔW_out| = 0.0)
  4. Interactive Enterprise ROI & Throughput Calculator (Self-Hosted Aligned 3.35B vs. Closed-Source Frontier APIs)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Support running either inside `hf_space/` (with bundled `results/`) or inside `experiment_02_aya_fire_lexsistack/hf_space/`
RESULTS_DIR = HERE / "results" if (HERE / "results" / "audit_ledger.json").exists() else HERE.parent / "results"
ARTICLE_PATH = HERE / "HF_ARTICLE.md" if (HERE / "HF_ARTICLE.md").exists() else HERE.parent / "HF_ARTICLE.md"


def _load_json(filename: str) -> Dict[str, Any]:
    p = RESULTS_DIR / filename
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_triaxial_dataframe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ledger = _load_json("audit_ledger.json")
    m = ledger.get("measured_tiny_aya_fire", {})
    ref = ledger.get("gemma4_article_reference", {}) or ledger

    rows: List[Dict[str, Any]] = []
    suite_specs = [
        ("Axis A: Harmful Refusal (↑)", "harmbench_refusal_pct", "HarmBench Direct (n=200)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "harmbench_multilingual_refusal_pct", "HarmBench Multilingual (6 langs)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "strongreject_refusal_pct", "StrongREJECT Adversarial (n=313)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "indicsafe_refusal_pct", "IndicSafe (12 Indic langs, n=3,960)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "sarvam_indic_safety_refusal_pct", "IndicSafetyBench (23 langs, n=3,374)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "banglasafe_refusal_pct", "BanglaSafe (BN/Banglish, n=879)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "multijail_refusal_pct", "MultiJail (10 langs, n=3,150)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "fortress_refusal_pct", "FORTRESS National Security (n=500)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "rmcbench_refusal_pct", "RMCBench Malicious Code (n=473)", "Refusal % (↑)"),
        ("Axis A: Harmful Refusal (↑)", "cyber_offensive_refusal_pct", "CyberDualUse Offensive", "Refusal % (↑)"),
        ("Axis B: Benign Over-Refusal (↓)", "or_bench_hard_refusal_pct", "OR-Bench-Hard-1k (n=1,319)", "Over-Refusal % (↓)"),
        ("Axis B: Benign Over-Refusal (↓)", "indicsafe_benign_refusal_pct", "IndicSafe Benign Control (n=1,500)", "Over-Refusal % (↓)"),
        ("Axis B: Benign Over-Refusal (↓)", "xstest_safe_refusal_pct", "XSTest Safe Triggers (n=250)", "Over-Refusal % (↓)"),
        ("Axis C: Knowledge Retention (↔)", "wmdp_cyber_acc", "WMDP-Cyber MCQ (n=1,987)", "Accuracy % (↔)"),
        ("Axis C: Knowledge Retention (↔)", "wmdp_bio_acc", "WMDP-Bio MCQ (n=1,273)", "Accuracy % (↔)"),
        ("Axis C: Knowledge Retention (↔)", "wmdp_chem_acc", "WMDP-Chem MCQ (n=408)", "Accuracy % (↔)"),
    ]

    base_m = m.get("base", {})
    ml_m = m.get("multilingual_repair", {})
    cy_m = m.get("cybersecurity_repair", {})

    for axis_label, key, label, metric_dir in suite_specs:
        if key in base_m or key in ml_m or key in cy_m:
            b_val = float(base_m.get(key, 0.0))
            m1_val = float(ml_m.get(key, b_val))
            m2_val = float(cy_m.get(key, b_val))
            best_delta = (m1_val - b_val) if "multilingual" in key or "indic" in key or "bangla" in key or "multijail" in key else (m2_val - b_val)
            rows.append(
                {
                    "Triaxial Dimension": axis_label,
                    "Benchmark Suite": label,
                    "Metric": metric_dir,
                    "Base (Before)": f"{b_val:.1f}%",
                    "Repair 1: Multilingual (C-ΔΘ)": f"{m1_val:.1f}%",
                    "Repair 2: Cybersecurity (C-ΔΘ)": f"{m2_val:.1f}%",
                    "Shift (Δ pp)": f"{best_delta:+.1f} pp",
                }
            )

    df_measured = pd.DataFrame(rows)

    # 13-Test Risk Map Table
    risk_rows = ref.get("risk_map_13_tests", [])
    df_risk = pd.DataFrame(risk_rows) if risk_rows else pd.DataFrame()
    return df_measured, df_risk


def plot_triaxial_chart() -> plt.Figure:
    ledger = _load_json("audit_ledger.json")
    m = ledger.get("measured_tiny_aya_fire", {})
    base_m = m.get("base", {})
    ckpts = [
        ("base", "Base (Before)", "#78909c"),
        ("multilingual_repair", "Repair 1: Multilingual (C-ΔΘ)", "#1e88e5"),
        ("cybersecurity_repair", "Repair 2: Cybersecurity (C-ΔΘ)", "#43a047"),
    ]

    axis_a_keys = [
        k
        for k in [
            "harmbench_refusal_pct",
            "harmbench_multilingual_refusal_pct",
            "strongreject_refusal_pct",
            "multijail_refusal_pct",
            "indicsafe_refusal_pct",
            "sarvam_indic_safety_refusal_pct",
            "banglasafe_refusal_pct",
            "fortress_refusal_pct",
            "rmcbench_refusal_pct",
            "cyber_offensive_refusal_pct",
        ]
        if k in base_m
    ]
    axis_b_keys = [
        k for k in ["or_bench_hard_refusal_pct", "indicsafe_benign_refusal_pct", "xstest_safe_refusal_pct"] if k in base_m
    ]
    axis_c_keys = [k for k in ["wmdp_cyber_acc", "wmdp_bio_acc", "wmdp_chem_acc"] if k in base_m]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 11.0), dpi=130)
    panels = [
        (axes[0], axis_a_keys, "Axis A — Harmful Refusal Rate % (Higher is Better ↑)"),
        (axes[1], axis_b_keys, "Axis B — Benign Over-Refusal Rate % (Lower is Better ↓)"),
        (axes[2], axis_c_keys, "Axis C — Hazardous Knowledge Preservation WMDP Accuracy % (Preserve ↔)"),
    ]
    for ax, keys, title in panels:
        if not keys:
            ax.set_visible(False)
            continue
        x = np.arange(len(keys))
        w = 0.26
        for idx, (ck_key, label, col) in enumerate(ckpts):
            vals = [float(m.get(ck_key, {}).get(k, 0.0)) for k in keys]
            rects = ax.bar(x + (idx - 1) * w, vals, width=w, label=label, color=col, edgecolor="#263238", linewidth=0.5)
            for r in rects:
                ax.text(
                    r.get_x() + r.get_width() / 2,
                    r.get_height() + 1.2,
                    f"{r.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [k.replace("_refusal_pct", "").replace("_acc", "") for k in keys], rotation=18, ha="right", fontsize=9
        )
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", ncol=3, fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_circuit_attribution() -> plt.Figure:
    nodes_data = _load_json("circuit_nodes.json")
    all_mlp = nodes_data.get("all_mlp_scores", {})
    target_layers = set(nodes_data.get("discovered_target_layers", []))

    layers = list(range(36))
    scores = [float(all_mlp.get(f"MLP {l}", 0.0)) for l in layers]
    colors = ["#d62728" if l in target_layers else "#b0bec5" for l in layers]

    fig, ax = plt.subplots(figsize=(12.5, 4.0), dpi=130)
    ax.bar(layers, scores, color=colors, edgecolor="#263238", linewidth=0.5)
    ax.set_title(
        "CircuitKIT EAP-IG Safety Circuit Attribution Across All 36 Transformer Layers (CohereLabs/tiny-aya-fire)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Transformer Layer Index (0–35)", fontsize=10.5)
    ax.set_ylabel("|EAP-IG Attribution Score|", fontsize=10.5)
    ax.set_xticks(layers)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        handles=[
            Patch(
                facecolor="#d62728",
                edgecolor="#263238",
                label=f"Selected Top-11 Target Circuit Layers ({sorted(target_layers)})",
            ),
            Patch(facecolor="#b0bec5", edgecolor="#263238", label="Frozen / Untouched Layers (max |ΔW| = 0.0)"),
        ],
        loc="upper right",
    )
    plt.tight_layout()
    return fig


def build_defensive_tasks_df() -> pd.DataFrame:
    ledger = _load_json("audit_ledger.json")
    m = ledger.get("measured_tiny_aya_fire", {})
    llm = ledger.get("llm_judge_scores") or {}

    base_tasks = llm.get("base", {}).get("defensive_10_tasks_grades") or m.get("base", {}).get("defensive_tasks", [])
    ml_map = {
        t["id"]: t
        for t in (
            llm.get("multilingual_repair", {}).get("defensive_10_tasks_grades")
            or m.get("multilingual_repair", {}).get("defensive_tasks", [])
        )
    }
    cy_map = {
        t["id"]: t
        for t in (
            llm.get("cybersecurity_repair", {}).get("defensive_10_tasks_grades")
            or m.get("cybersecurity_repair", {}).get("defensive_tasks", [])
        )
    }

    rows = []
    for t in base_tasks:
        tid = t["id"]
        rows.append(
            {
                "Task ID": tid,
                "Defensive Security Engineering Task": t["title"],
                "Base Grade": t["grade"],
                "Repair 1 (Multilingual) Grade": ml_map.get(tid, {}).get("grade", t["grade"]),
                "Repair 2 (Cybersecurity) Grade": cy_map.get(tid, {}).get("grade", t["grade"]),
                "Response Preview (Repaired)": cy_map.get(tid, {}).get("response_preview", t.get("response_preview", "")),
            }
        )
    return pd.DataFrame(rows)


def get_transcript_examples() -> Dict[str, Dict[str, str]]:
    gen = _load_json("aya_fire_generations.json").get("transcripts", {})
    examples: Dict[str, Dict[str, str]] = {}

    for ckpt in ("base", "multilingual_repair", "cybersecurity_repair"):
        for sname, items in gen.get(ckpt, {}).items():
            for item in items[:6]:
                key = f"[{sname} | lang={item.get('lang', 'en')}] {item.get('prompt', '')[:90]}..."
                examples.setdefault(
                    key,
                    {
                        "suite": sname,
                        "lang": item.get("lang", "en"),
                        "prompt": item.get("prompt", ""),
                        "base": "",
                        "multilingual_repair": "",
                        "cybersecurity_repair": "",
                    },
                )
                examples[key][ckpt] = item.get("response", "")
    return examples


def compute_roi_table(
    monthly_requests_millions: float,
    avg_input_tokens: int,
    avg_output_tokens: int,
    gpu_hourly_usd: float,
    tokens_per_sec: int,
) -> Tuple[pd.DataFrame, str, plt.Figure]:
    reqs = monthly_requests_millions * 1_000_000.0
    in_tokens = reqs * avg_input_tokens
    out_tokens = reqs * avg_output_tokens
    total_tokens = in_tokens + out_tokens

    gpu_hours = total_tokens / max(tokens_per_sec * 3600.0, 1.0)
    min_gpus = max(1, math.ceil(gpu_hours / 720.0))
    self_host_compute_usd = gpu_hours * gpu_hourly_usd
    reserved_compute_usd = gpu_hours * (gpu_hourly_usd * 0.56)

    models = [
        (
            "OpenAI GPT-4o (Closed API)",
            "$2.50 / 1M in · $10.00 / 1M out",
            0.0,
            (in_tokens / 1e6) * 2.50 + (out_tokens / 1e6) * 10.00,
            0.0,
        ),
        (
            "Anthropic Claude 3.5 Sonnet (Closed API)",
            "$3.00 / 1M in · $15.00 / 1M out",
            0.0,
            (in_tokens / 1e6) * 3.00 + (out_tokens / 1e6) * 15.00,
            0.0,
        ),
        (
            "Cohere Command R+ (Closed API)",
            "$2.50 / 1M in · $10.00 / 1M out",
            0.0,
            (in_tokens / 1e6) * 2.50 + (out_tokens / 1e6) * 10.00,
            0.0,
        ),
        (
            "OpenAI GPT-4o-mini + Moderation API",
            "$0.15 / 1M in · $0.60 / 1M out + Guard",
            0.0,
            (in_tokens / 1e6) * 0.15 + (out_tokens / 1e6) * 0.60,
            reqs * 0.000045,
        ),
        (
            "Self-Hosted 3.35B + Full SFT/DPO + 7B Guardrail Sidecar",
            f"{gpu_hours:.0f} GPU-hrs Model + {gpu_hours:.0f} GPU-hrs 7B Guard",
            480.0,
            self_host_compute_usd,
            self_host_compute_usd,
        ),
        (
            "Ours: Self-Hosted Tiny-Aya-Fire (3.35B) + Lexsi C-ΔΘ Repair",
            f"{gpu_hours:.0f} GPU-hrs @ ${gpu_hourly_usd:.2f}/hr (0ms Guard Tax)",
            0.01,
            self_host_compute_usd,
            0.0,
        ),
    ]

    gpt4o_total = models[0][3] + models[0][4]
    ours_total = max(self_host_compute_usd, 0.01)

    rows = []
    chart_labels = []
    chart_totals = []
    for name, billing, align_cost, infer_cost, guard_cost in models:
        tot = infer_cost + guard_cost
        savings_pct = 100.0 * (1.0 - tot / max(gpt4o_total, 1e-6))
        mult = gpt4o_total / max(tot, 1e-6)
        rows.append(
            {
                "Deployment Architecture": name,
                "Billing / Throughput Formula": billing,
                "One-Time Alignment Cost": f"${align_cost:,.2f}",
                "Monthly Inference Cost": f"${infer_cost:,.2f}",
                "External Guardrail Cost": f"${guard_cost:,.2f}",
                "Total Monthly Cost": f"${tot:,.2f}",
                "Savings vs. GPT-4o": f"{savings_pct:+.1f}% ({mult:.1f}x cheaper)" if tot <= gpt4o_total else f"{savings_pct:+.1f}%",
            }
        )
        chart_labels.append(name.split(" (")[0].replace("Ours: ", ""))
        chart_totals.append(tot)

    summary_md = (
        f"### 📊 Workload & ROI Summary\n"
        f"- **Monthly Volume**: `{monthly_requests_millions:.1f}M requests/mo` = **`{total_tokens / 1e9:.2f} Billion tokens/mo`** "
        f"(`{in_tokens / 1e9:.2f}B` input + `{out_tokens / 1e9:.2f}B` output)\n"
        f"- **GPU Runtime Required (`tiny-aya-fire` 3.35B `bfloat16` @ `{tokens_per_sec:,} tok/s`)**: **`{gpu_hours:,.1f} GPU hours/month`** "
        f"(~`{min_gpus}` GPU instance(s) active)\n"
        f"- **Monthly Self-Hosted Cost (`C-ΔΘ` Repaired)**: **`${self_host_compute_usd:,.2f}/mo` on-demand** "
        f"(`~${reserved_compute_usd:,.2f}/mo` reserved) with **$0 external guardrail GPU cost** and **$0.01 one-time alignment cost** (`45s` on 1× L4 GPU)\n"
        f"- **Net Annual Savings vs. `GPT-4o` / `Command R+`**: **`${(gpt4o_total - ours_total) * 12:,.2f} / year` (`{gpt4o_total / ours_total:.1f}×` cheaper)**"
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=130)
    colors = ["#ef5350", "#ab47bc", "#ffa726", "#29b6f6", "#78909c", "#2e7d32"]
    y_pos = np.arange(len(chart_labels))
    bars = ax.barh(y_pos, chart_totals, color=colors, edgecolor="#263238", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Total Monthly Cost (USD — Lower is Better)", fontsize=10.5)
    ax.set_title(
        f"Monthly Production TCO at {monthly_requests_millions:.1f}M Requests/Mo ({total_tokens / 1e9:.2f}B Tokens/Mo)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    max_val = max(chart_totals) if chart_totals else 1.0
    ax.set_xlim(0, max_val * 1.22)
    for b in bars:
        w = b.get_width()
        ax.text(w + max_val * 0.015, b.get_y() + b.get_height() / 2, f"${w:,.0f}/mo", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()

    return pd.DataFrame(rows), summary_md, fig


def build_app() -> gr.Blocks:
    df_measured, df_risk = build_triaxial_dataframe()
    df_def = build_defensive_tasks_df()
    examples = get_transcript_examples()
    example_keys = list(examples.keys())
    prov = _load_json("lexsi_provenance.json")
    circuit_html_path = RESULTS_DIR / "circuit_graph.html"
    circuit_html = circuit_html_path.read_text(encoding="utf-8") if circuit_html_path.exists() else "<p>Run pipeline to generate circuit_graph.html</p>"
    article_md = ARTICLE_PATH.read_text(encoding="utf-8") if ARTICLE_PATH.exists() else "# HF Article not found"

    with gr.Blocks(title="Tiny-Aya-Fire Targeted Model Repair | Lexsi Stack") as demo:
        gr.Markdown(
            """
            # 🛡️ Targeted Model Repair on `CohereLabs/tiny-aya-fire` (3.35B) — Lexsi Alignment & Safety Stack
            **Offline Circuit-Restricted Weight Arithmetic ($C\\text{-}\\Delta\\Theta$) across 11 EAP-IG Discovered MLP Layers (`max |ΔW_outside_circuit| = 0.0`)**
            - **Base Model**: [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (`Cohere2ForCausalLM`, 3.35B parameters, 36 layers, `bfloat16`)
            - **Stack**: `CuratorKIT` (Disjoint Curation) $\\rightarrow$ `CircuitKIT` (EAP-IG Attribution) $\\rightarrow$ `SafeTune` ($C\\text{-}\\Delta\\Theta$ Weight Repair) $\\rightarrow$ `AuditKIT` (Triaxial Audit & Two-Judge Panel)
            """
        )

        with gr.Tabs():
            with gr.TabItem("📈 1. Before-and-After Triaxial Metrics"):
                gr.Markdown(
                    "### Triaxial Evaluation: Harmful Refusal ($\\uparrow$) · Benign Over-Refusal ($\\downarrow$) · Hazardous Knowledge Retention ($\\leftrightarrow$)"
                )
                gr.Plot(value=plot_triaxial_chart, label="Triaxial Before vs. After Comparison")
                gr.Markdown("#### Measured Before-and-After Metrics on `CohereLabs/tiny-aya-fire`")
                gr.Dataframe(value=df_measured, interactive=False)
                gr.Markdown("#### Ten Canonical Defensive Cybersecurity Engineering Tasks (`E/O/C/R/F` 5-Step Actionability Scale)")
                gr.Dataframe(value=df_def, interactive=False)
                if not df_risk.empty:
                    gr.Markdown("#### Lexsi 13-Test Shared Audit Risk Map Reference")
                    gr.Dataframe(value=df_risk, interactive=False)

            with gr.TabItem("💬 2. Side-by-Side Transcript Explorer (Before vs. After)"):
                gr.Markdown(
                    "### Compare Actual Model Outputs: Uncorrected Base vs. Multilingual Repair ($C\\text{-}\\Delta\\Theta$) vs. Cybersecurity Repair ($C\\text{-}\\Delta\\Theta$)"
                )
                if example_keys:
                    selector = gr.Dropdown(
                        choices=example_keys,
                        value=example_keys[0],
                        label="Select Held-Out Benchmark Prompt (Multilingual Indic, Jailbreak, Benign Trigger, or Defensive Cyber Task)",
                    )
                    prompt_box = gr.Textbox(label="Prompt Text", value=examples[example_keys[0]]["prompt"], lines=3, interactive=False)
                    with gr.Row():
                        base_box = gr.Textbox(
                            label="🔴 Base CohereLabs/tiny-aya-fire (Before Repair)",
                            value=examples[example_keys[0]]["base"],
                            lines=8,
                            interactive=False,
                        )
                        ml_box = gr.Textbox(
                            label="🔵 Repair 1: Multilingual C-ΔΘ (6 Langs Consolidated)",
                            value=examples[example_keys[0]]["multilingual_repair"],
                            lines=8,
                            interactive=False,
                        )
                        cy_box = gr.Textbox(
                            label="🟢 Repair 2: Cybersecurity C-ΔΘ (AU Suppressed, Defensive E/O Preserved)",
                            value=examples[example_keys[0]]["cybersecurity_repair"],
                            lines=8,
                            interactive=False,
                        )

                    def _update_example(k: str) -> Tuple[str, str, str, str]:
                        ex = examples.get(k, {})
                        return (
                            ex.get("prompt", ""),
                            ex.get("base", ""),
                            ex.get("multilingual_repair", ""),
                            ex.get("cybersecurity_repair", ""),
                        )

                    selector.change(_update_example, inputs=[selector], outputs=[prompt_box, base_box, ml_box, cy_box])

            with gr.TabItem("🔬 3. Mechanistic Circuit & Cryptographic Provenance"):
                gr.Markdown(
                    "### EAP-IG Safety Circuit Localization Across All 36 Layers (`CircuitKIT` + `SafeTune.interpret`)"
                )
                gr.Plot(value=plot_circuit_attribution, label="Per-Layer MLP EAP-IG Attribution")
                gr.Markdown("### Interactive D3 Causal Circuit Graph (`circuit_graph.html`)")
                gr.HTML(value=circuit_html)
                gr.Markdown("### Cryptographic Provenance & Non-Circuit Weight Invariance (`lexsi_provenance.json`)")
                gr.JSON(value=prov)

            with gr.TabItem("💰 4. Enterprise ROI & Throughput Calculator"):
                gr.Markdown(
                    "### Production ROI Comparison: Self-Hosted Aligned `tiny-aya-fire` (3.35B) vs. Closed-Source Frontier Models"
                )
                with gr.Row():
                    sl_reqs = gr.Slider(0.5, 50.0, value=10.0, step=0.5, label="Monthly Request Volume (Millions of Requests)")
                    sl_in = gr.Slider(50, 2000, value=350, step=50, label="Avg Input Tokens / Request")
                    sl_out = gr.Slider(50, 1000, value=250, step=25, label="Avg Output Tokens / Request")
                with gr.Row():
                    sl_gpu = gr.Slider(0.35, 3.50, value=0.80, step=0.05, label="GPU Hourly Price ($/hr — e.g. $0.80 L4, $1.00 A10G, $2.40 A100)")
                    sl_tps = gr.Slider(400, 5000, value=2200, step=100, label="Estimated Throughput (Tokens/sec per GPU in bfloat16 vLLM)")

                init_df, init_md, init_fig = compute_roi_table(10.0, 350, 250, 0.80, 2200)
                roi_md = gr.Markdown(value=init_md)
                roi_plot = gr.Plot(value=init_fig)
                roi_table = gr.Dataframe(value=init_df, interactive=False)

                for ctrl in (sl_reqs, sl_in, sl_out, sl_gpu, sl_tps):
                    ctrl.change(
                        compute_roi_table,
                        inputs=[sl_reqs, sl_in, sl_out, sl_gpu, sl_tps],
                        outputs=[roi_table, roi_md, roi_plot],
                    )

            with gr.TabItem("📄 5. Full HF Article & Technical Write-Up"):
                gr.Markdown(value=article_md)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
