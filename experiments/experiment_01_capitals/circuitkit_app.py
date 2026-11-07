"""
CircuitKIT Web Studio - Interactive UI for Circuit Discovery, Evaluation & Pruning.
Built on top of Lexsi Labs' CircuitKIT and Streamlit.
"""

import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch

import yaml
import circuitkit as ck
from circuitkit import Pipeline
from circuitkit.circuit import Circuit
from circuitkit.tasks.yaml_loader import YAMLTaskLoader
from circuitkit.tasks.registry import register_task, is_task_registered

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CircuitKIT Studio | Lexsi Labs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B0A 0%, #FF8A00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        gap: 4px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        border-bottom: 2px solid #FF4B0A !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "circuit" not in st.session_state:
    st.session_state.circuit = None
if "scores_dict" not in st.session_state:
    st.session_state.scores_dict = None
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None
if "viz_html" not in st.session_state:
    st.session_state.viz_html = None
if "active_model" not in st.session_state:
    st.session_state.active_model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt2"
if "logs" not in st.session_state:
    st.session_state.logs = []
if "custom_tasks" not in st.session_state:
    st.session_state.custom_tasks = ["custom_capitals"] if os.path.exists("custom_tasks_demo/capitals.yaml") else []
if "active_task" not in st.session_state:
    st.session_state.active_task = "ioi"


def add_log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {msg}")


# -----------------------------------------------------------------------------
# Model Caching Helper
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HookedTransformer model weights...")
def get_model(model_name: str):
    return ck.load_model(model_name)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Lexsi-Labs/CircuitKIT/main/docs/assets/circuitkit-logo-white.png", width=220)
    st.caption("Mechanistic Interpretability Toolkit")
    st.markdown("---")

    st.subheader("⚙️ Target Model & Task")
    model_choice = st.selectbox(
        "Select Model",
        ["gpt2", "gpt2-medium", "facebook/opt-125m", "Custom (Local or Hub)"],
        index=0
    )
    if model_choice == "Custom (Local or Hub)":
        target_model = st.text_input("Enter Hub repo ID or local checkpoint dir", value="gpt2")
    else:
        target_model = model_choice
    st.session_state.model_name = target_model

    tasks_list = ck.list_tasks() if hasattr(ck, "list_tasks") else ["ioi", "greater_than", "gender_bias", "sva"]
    task_mode = st.radio("Task Source", ["Built-in Tasks", "🎯 Custom Task"], horizontal=True)
    
    if task_mode == "Built-in Tasks":
        default_task_idx = tasks_list.index("ioi") if "ioi" in tasks_list else 0
        selected_task = st.selectbox("Task Benchmark", tasks_list, index=default_task_idx)
    else:
        if st.session_state.custom_tasks:
            # Register if not already in registry
            for ct in st.session_state.custom_tasks:
                yaml_p = Path(f"custom_tasks_demo/{ct}.yaml")
                if yaml_p.exists() and not is_task_registered(ct):
                    try:
                        spec = YAMLTaskLoader.load(yaml_p)
                        register_task(spec)
                    except Exception:
                        pass
            selected_task = st.selectbox("Select Custom Task", st.session_state.custom_tasks)
        else:
            st.info("No custom task registered yet. Create one in the '🎯 Custom Task Builder' tab!")
            selected_task = "ioi"
    
    st.session_state.active_task = selected_task

    st.markdown("---")
    st.subheader("🔬 Discovery Hyperparameters")
    
    algorithm = st.selectbox(
        "Algorithm",
        ["eap-ig", "eap", "acdc", "edge-attribution", "taylor"],
        index=0,
        help="EAP-IG: Edge Attribution Patching with Integrated Gradients"
    )
    
    sparsity = st.slider("Target Sparsity", min_value=0.05, max_value=0.90, value=0.20, step=0.05,
                         help="Proportion of model components/edges to prune (0.2 = keep top 80%)")
    
    level = st.radio("Component Level", ["node", "edge"], horizontal=True, index=0)
    scope = st.selectbox("Scope", ["both", "heads", "mlp"], index=0)
    
    col_a, col_b = st.columns(2)
    with col_a:
        n_examples = st.number_input("Samples", min_value=4, max_value=256, value=8, step=4)
    with col_b:
        ig_steps = st.number_input("IG Steps", min_value=1, max_value=10, value=2, step=1)

    st.markdown("---")
    st.caption("⚡ CircuitKIT runs on CPU or GPU without extra config.")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
col_title, col_quick = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-title">CircuitKIT Interactive Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Discover, evaluate, visualize, and structurally intervene on transformer neural circuits.</div>', unsafe_allow_html=True)

with col_quick:
    # Quick loader for cached demo
    if os.path.exists("circuit_discovery_results.pt") and os.path.exists("circuit_discovery_results_scores.json"):
        if st.button("⚡ Load Cached IOI Circuit", use_container_width=True, type="secondary"):
            try:
                c = ck.load_scores("circuit_discovery_results.pt")
                with open("circuit_discovery_results_scores.json", "r") as f:
                    scores_data = json.load(f)
                st.session_state.circuit = c
                st.session_state.scores_dict = scores_data
                if os.path.exists("circuit_viz.html"):
                    with open("circuit_viz.html", "r") as vf:
                        st.session_state.viz_html = vf.read()
                if os.path.exists("circuit_eval_results.json"):
                    with open("circuit_eval_results.json", "r") as ef:
                        st.session_state.eval_data = json.load(ef)
                add_log("Loaded cached circuit discovery results and evaluation from disk.")
                st.toast("Loaded cached GPT-2 IOI circuit & evaluation!", icon="✅")
            except Exception as e:
                st.error(f"Error loading cached circuit: {e}")

# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_pipeline, tab_custom, tab_viz, tab_components, tab_faithfulness, tab_inference, tab_docs = st.tabs([
    "🚀 Pipeline & Discovery",
    "🎯 Custom Task Builder",
    "🔍 Interactive Graph",
    "📊 Component Scores",
    "🛡️ Faithfulness Evaluation",
    "🎮 Prompt Playground",
    "📖 Guide & Code"
])


# =============================================================================
# TAB 1: Pipeline & Discovery
# =============================================================================
with tab_pipeline:
    st.markdown("### 1. Circuit Discovery Workflow")
    st.markdown("Run the complete interpretability loop: isolate task-driving subgraphs, measure faithfulness, and export pruned weights.")

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        run_discovery_btn = st.button("1. 🔎 Discover Circuit", use_container_width=True, type="primary")
    with c2:
        run_eval_btn = st.button("2. 📊 Evaluate Faithfulness", use_container_width=True)
    with c3:
        run_prune_btn = st.button("3. ✂️ Prune Model", use_container_width=True)
    with c4:
        run_export_btn = st.button("4. 💾 Export Checkpoint", use_container_width=True)

    if run_discovery_btn:
        with st.status("Executing Circuit Discovery...", expanded=True) as status:
            try:
                add_log(f"Loading model '{target_model}'...")
                st.write(f"📥 Loading HookedTransformer for `{target_model}`...")
                model = get_model(target_model)
                st.session_state.active_model = model

                add_log(f"Running discovery: task={selected_task}, algorithm={algorithm}, sparsity={sparsity}...")
                st.write(f"🔬 Discovering circuit with `{algorithm}` on `{selected_task}` ({n_examples} examples)...")
                
                output_artifact = "circuit_discovery_results.pt"
                circuit = ck.discover(
                    model,
                    selected_task,
                    algorithm=algorithm,
                    level=level,
                    n_examples=int(n_examples),
                    batch_size=2,
                    ig_steps=int(ig_steps),
                    sparsity=float(sparsity),
                    scope=scope,
                    output_path=output_artifact
                )
                
                st.session_state.circuit = circuit
                add_log(f"Discovery complete. Discovered {len(circuit)} nodes.")
                
                # Load scores side-car JSON
                scores_json_path = "circuit_discovery_results_scores.json"
                if os.path.exists(scores_json_path):
                    with open(scores_json_path, "r") as f:
                        st.session_state.scores_dict = json.load(f)

                # Generate D3 visualization
                st.write("🎨 Rendering interactive D3 circuit graph...")
                ck.visualize_circuit(circuit, mode="graph", output="circuit_viz.html")
                if os.path.exists("circuit_viz.html"):
                    with open("circuit_viz.html", "r") as vf:
                        st.session_state.viz_html = vf.read()

                status.update(label=f"Circuit Discovery Finished! ({len(circuit)} nodes retained)", state="complete")
                st.toast("Discovery finished successfully!", icon="🎉")
            except Exception as e:
                status.update(label="Discovery Failed", state="error")
                st.error(f"Error during discovery: {str(e)}")
                add_log(f"ERROR: {str(e)}")

    if run_eval_btn:
        if st.session_state.circuit is None:
            st.warning("Please discover or load a circuit first before evaluating.")
        else:
            with st.status("Running Multi-Pillar Faithfulness Evaluation...", expanded=True) as status:
                try:
                    model = get_model(target_model)
                    st.write("Evaluating causal patching & ablation stability...")
                    add_log("Evaluating faithfulness (patching, ablation)...")
                    report = ck.faithfulness(
                        model,
                        st.session_state.circuit,
                        selected_task,
                        pillars=["patching", "ablation"],
                        n_examples=int(n_examples)
                    )
                    st.session_state.eval_report = report
                    status.update(label="Evaluation Finished!", state="complete")
                    st.toast("Evaluation complete!", icon="✅")
                    add_log(f"Evaluation complete: Patching={report.patching_score:.4f}, Ablation={report.ablation_score:.4f}")
                except Exception as e:
                    status.update(label="Evaluation Failed", state="error")
                    st.error(f"Evaluation error: {str(e)}")

    if run_prune_btn:
        if st.session_state.circuit is None:
            st.warning("Please discover or load a circuit first.")
        else:
            with st.spinner("Pruning model down to circuit..."):
                try:
                    model = get_model(target_model)
                    pruned = ck.prune(model, st.session_state.circuit, sparsity=sparsity, scope=scope)
                    st.success("Successfully created pruned model in memory! Original model remains unperturbed.")
                    add_log("Model pruned successfully.")
                except Exception as e:
                    st.error(f"Pruning error: {e}")

    if run_export_btn:
        if st.session_state.circuit is None:
            st.warning("Please discover or load a circuit first.")
        else:
            with st.spinner("Exporting HuggingFace checkpoint & Lexsi provenance..."):
                try:
                    model = get_model(target_model)
                    export_dir = "./exported_checkpoint"
                    path = ck.export_checkpoint(model, st.session_state.circuit, export_dir)
                    st.success(f"Checkpoint exported to `{path}`! Includes `lexsi_provenance.json`.")
                    add_log(f"Checkpoint exported to {path}")
                except Exception as e:
                    st.error(f"Export error: {e}")

    st.markdown("---")
    # Quick Overview Cards
    if st.session_state.circuit is not None:
        c = st.session_state.circuit
        c_cols = st.columns(4)
        with c_cols[0]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(c)}</div>
                <div class="metric-label">Nodes in Circuit</div>
            </div>""", unsafe_allow_html=True)
        with c_cols[1]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{selected_task.upper()}</div>
                <div class="metric-label">Benchmark Task</div>
            </div>""", unsafe_allow_html=True)
        with c_cols[2]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{sparsity * 100:.0f}%</div>
                <div class="metric-label">Pruning Sparsity</div>
            </div>""", unsafe_allow_html=True)
        with c_cols[3]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{target_model}</div>
                <div class="metric-label">Model Architecture</div>
            </div>""", unsafe_allow_html=True)

    # Activity Log
    with st.expander("📜 Activity Logs", expanded=False):
        if st.session_state.logs:
            st.code("\n".join(st.session_state.logs), language="bash")
        else:
            st.info("No activity logged yet.")


# =============================================================================
# TAB 1B: Custom Task Builder
# =============================================================================
with tab_custom:
    st.markdown("### 🎯 Custom Task Builder (Bring Your Own Data)")
    st.markdown("""
    CircuitKIT supports any custom task without writing code. Define your contrastive prompt pairs below,
    upload a dataset (`.csv`), or supply a task YAML configuration.
    """)

    os.makedirs("custom_tasks", exist_ok=True)
    custom_mode = st.radio("Task Input Mode", ["⚡ Interactive Prompt Editor", "📁 Upload Dataset (CSV)", "📝 Declarative YAML"], horizontal=True)

    if custom_mode == "⚡ Interactive Prompt Editor":
        col_tname, col_metric = st.columns([2, 1])
        with col_tname:
            new_task_name = st.text_input("Task Identifier (lowercase, no spaces)", value="custom_world_capitals")
        with col_metric:
            metric_type = st.selectbox("Causal Metric", ["logit_diff", "kl", "accuracy"], index=0)

        st.caption("Edit prompt pairs in the table below or add new rows (Clean vs Corrupted / Counterfactual):")
        
        default_data = pd.DataFrame([
            {"prompt": "The capital of France is", "answer": " Paris", "corrupted_prompt": "The capital of Italy is", "corrupted_answer": " Rome"},
            {"prompt": "The capital of Spain is", "answer": " Madrid", "corrupted_prompt": "The capital of Germany is", "corrupted_answer": " Berlin"},
            {"prompt": "The capital of Japan is", "answer": " Tokyo", "corrupted_prompt": "The capital of China is", "corrupted_answer": " Beijing"},
            {"prompt": "The capital of England is", "answer": " London", "corrupted_prompt": "The capital of Egypt is", "corrupted_answer": " Cairo"},
        ])
        
        edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

        if st.button("💾 Save & Register Task", type="primary"):
            if not new_task_name.strip():
                st.error("Please enter a valid task name.")
            elif len(edited_df) < 2:
                st.error("Please provide at least 2 contrastive prompt rows.")
            else:
                csv_path = f"custom_tasks/{new_task_name}.csv"
                yaml_path = f"custom_tasks/{new_task_name}.yaml"
                
                edited_df.to_csv(csv_path, index=False)
                
                yaml_dict = {
                    "name": new_task_name,
                    "source": {"type": "csv", "path": csv_path},
                    "schema": {
                        "prompt": "prompt",
                        "answer": "answer",
                        "corrupted_prompt": "corrupted_prompt",
                        "corrupted_answer": "corrupted_answer"
                    },
                    "metric": metric_type
                }
                with open(yaml_path, "w") as yf:
                    yaml.dump(yaml_dict, yf, default_flow_style=False)

                try:
                    task_spec = YAMLTaskLoader.load(Path(yaml_path))
                    register_task(task_spec)
                    if new_task_name not in st.session_state.custom_tasks:
                        st.session_state.custom_tasks.append(new_task_name)
                    st.session_state.active_task = new_task_name
                    st.success(f"Task `{new_task_name}` successfully registered in CircuitKIT! You can now select it in the sidebar.")
                    st.toast(f"Custom task '{new_task_name}' registered!", icon="🎯")
                    add_log(f"Registered custom task '{new_task_name}' ({len(edited_df)} samples)")
                except Exception as ex:
                    st.error(f"Failed to load/register task: {ex}")

    elif custom_mode == "📁 Upload Dataset (CSV)":
        up_col1, up_col2 = st.columns([2, 1])
        with up_col1:
            up_task_name = st.text_input("Custom Task Name", value="my_uploaded_task")
        with up_col2:
            up_metric = st.selectbox("Evaluation Metric", ["logit_diff", "kl", "accuracy"], index=0, key="up_metric")
            
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            up_df = pd.read_csv(uploaded_file)
            st.write(f"Preview (Total {len(up_df)} rows):")
            st.dataframe(up_df.head(5), use_container_width=True)

            cols = list(up_df.columns)
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                col_prompt = st.selectbox("Clean Prompt Column", cols, index=0 if len(cols) > 0 else 0)
            with mcol2:
                col_answer = st.selectbox("Clean Answer Column", cols, index=min(1, len(cols)-1))
            with mcol3:
                col_corrupt = st.selectbox("Corrupted Prompt Column", cols, index=min(2, len(cols)-1))
            with mcol4:
                col_c_ans = st.selectbox("Corrupted Answer Column", cols, index=min(3, len(cols)-1))

            if st.button("🚀 Register Uploaded Dataset", type="primary"):
                saved_csv = f"custom_tasks/{up_task_name}.csv"
                saved_yaml = f"custom_tasks/{up_task_name}.yaml"
                up_df.to_csv(saved_csv, index=False)
                
                yaml_dict = {
                    "name": up_task_name,
                    "source": {"type": "csv", "path": saved_csv},
                    "schema": {
                        "prompt": col_prompt,
                        "answer": col_answer,
                        "corrupted_prompt": col_corrupt,
                        "corrupted_answer": col_c_ans
                    },
                    "metric": up_metric
                }
                with open(saved_yaml, "w") as yf:
                    yaml.dump(yaml_dict, yf, default_flow_style=False)

                try:
                    task_spec = YAMLTaskLoader.load(Path(saved_yaml))
                    register_task(task_spec)
                    if up_task_name not in st.session_state.custom_tasks:
                        st.session_state.custom_tasks.append(up_task_name)
                    st.session_state.active_task = up_task_name
                    st.success(f"Custom dataset registered as task `{up_task_name}`!")
                    st.toast(f"Registered '{up_task_name}' successfully!", icon="✅")
                    add_log(f"Registered uploaded task '{up_task_name}' with {len(up_df)} samples.")
                except Exception as ex:
                    st.error(f"Error registering dataset: {ex}")

    elif custom_mode == "📝 Declarative YAML":
        st.caption("Paste or edit your CircuitKIT task YAML configuration below:")
        default_yaml = """name: custom_yaml_task
source:
  type: csv
  path: custom_tasks_demo/capitals.csv
schema:
  prompt: question
  answer: answer
  corrupted_prompt: counterfactual
  corrupted_answer: wrong_answer
metric: logit_diff
"""
        yaml_text = st.text_area("Task YAML", value=default_yaml, height=220)
        if st.button("Register from YAML", type="primary"):
            try:
                parsed = yaml.safe_load(yaml_text)
                t_name = parsed.get("name", "custom_task")
                save_path = f"custom_tasks/{t_name}.yaml"
                with open(save_path, "w") as f:
                    f.write(yaml_text)
                task_spec = YAMLTaskLoader.load(Path(save_path))
                register_task(task_spec)
                if t_name not in st.session_state.custom_tasks:
                    st.session_state.custom_tasks.append(t_name)
                st.session_state.active_task = t_name
                st.success(f"Task `{t_name}` registered and loaded!")
                st.toast(f"Task '{t_name}' registered!", icon="✅")
                add_log(f"Registered task '{t_name}' from raw YAML")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")


# =============================================================================
# TAB 2: Interactive D3 Graph Visualizer
# =============================================================================
with tab_viz:
    st.markdown("### 🔍 Interactive Circuit Graph")
    st.markdown("Interactive D3.js visualization generated by CircuitKIT. Hover over nodes to inspect attention heads and MLPs; adjust threshold to filter edge attributions.")

    if st.session_state.viz_html is not None:
        components.html(st.session_state.viz_html, height=720, scrolling=True)
    elif os.path.exists("circuit_viz.html"):
        with open("circuit_viz.html", "r") as f:
            html_content = f.read()
            components.html(html_content, height=720, scrolling=True)
    else:
        st.info("No visualization available yet. Run 'Discover Circuit' or click 'Load Cached IOI Circuit' above.")


# =============================================================================
# TAB 3: Component Scores
# =============================================================================
with tab_components:
    st.markdown("### 📊 Discovered Component Scores & Attributions")
    st.markdown("CircuitKIT ranks every component (Attention Head, MLP) by its causal contribution to the task.")

    if st.session_state.scores_dict and "node_scores" in st.session_state.scores_dict:
        node_scores = st.session_state.scores_dict["node_scores"]
        
        # Build DataFrame
        records = []
        for name, score in node_scores.items():
            if name.startswith("A"):
                parts = name[1:].split(".")
                layer = int(parts[0])
                head = int(parts[1]) if len(parts) > 1 else None
                comp_type = "Attention Head"
            elif name.startswith("MLP"):
                parts = name.split()
                layer = int(parts[1])
                head = None
                comp_type = "MLP Layer"
            else:
                layer = 0
                head = None
                comp_type = "Other"

            records.append({
                "Component": name,
                "Type": comp_type,
                "Layer": layer,
                "Head": head,
                "Importance Score": float(score)
            })

        df = pd.DataFrame(records).sort_values(by="Importance Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

        col_top, col_dist = st.columns(2)
        
        with col_top:
            st.subheader("Top 15 Most Critical Components")
            top15 = df.head(15)
            fig_bar = px.bar(
                top15,
                x="Importance Score",
                y="Component",
                orientation="h",
                color="Type",
                color_discrete_map={"Attention Head": "#FF4B0A", "MLP Layer": "#58A6FF"},
                title="Top Critical Components"
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), template="plotly_dark", height=420)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_dist:
            st.subheader("Attribution Distribution by Layer")
            layer_agg = df.groupby(["Layer", "Type"])["Importance Score"].sum().reset_index()
            fig_layer = px.bar(
                layer_agg,
                x="Layer",
                y="Importance Score",
                color="Type",
                barmode="group",
                color_discrete_map={"Attention Head": "#FF4B0A", "MLP Layer": "#58A6FF"},
                title="Total Attribution across Transformer Depth"
            )
            fig_layer.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_layer, use_container_width=True)

        st.subheader("Detailed Component Scores Table")
        st.dataframe(
            df[["Rank", "Component", "Type", "Layer", "Head", "Importance Score"]],
            use_container_width=True,
            height=300
        )

        # Download buttons
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            json_str = json.dumps(st.session_state.scores_dict, indent=2)
            st.download_button(
                label="📥 Download Scores (JSON)",
                data=json_str,
                file_name="circuit_scores.json",
                mime="application/json"
            )
        with dcol2:
            if os.path.exists("circuit_discovery_results.pt"):
                with open("circuit_discovery_results.pt", "rb") as pf:
                    st.download_button(
                        label="📥 Download Circuit Artifact (.pt)",
                        data=pf.read(),
                        file_name="circuit_artifact.pt",
                        mime="application/octet-stream"
                    )
    else:
        st.info("No score data available. Run 'Discover Circuit' or click 'Load Cached IOI Circuit'.")


# =============================================================================
# TAB 4: Faithfulness Evaluation
# =============================================================================
with tab_faithfulness:
    st.markdown("### 🛡️ Multi-Pillar Faithfulness Evaluation")
    st.markdown("""
    CircuitKIT evaluates whether the discovered circuit is **genuinely responsible** for model behavior using a rigorous multi-pillar protocol:
    * **Pillar 1 (Causal Patching)**: Measures how well the isolated circuit reconstructs clean output logits when restored in a corrupted run.
    * **Pillar 2 (Ablation Stability)**: Measures performance drop when the circuit is ablated/removed from the original model.
    """)

    eval_obj = st.session_state.eval_report
    eval_dict = getattr(st.session_state, "eval_data", None)

    if eval_obj is not None or eval_dict is not None:
        patch_score = eval_obj.patching_score if eval_obj else eval_dict.get("patching_score", 0.0)
        ablate_score = eval_obj.ablation_score if eval_obj else eval_dict.get("ablation_score", 0.0)
        
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{patch_score:.4f}</div>
                <div class="metric-label">Causal Patching Score (Pillar 1)</div>
            </div>""", unsafe_allow_html=True)
            st.caption("A score near 1.0 indicates the circuit alone carries sufficient information to reproduce the task.")

        with ecol2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{ablate_score:.4f}</div>
                <div class="metric-label">Ablation Drop Score (Pillar 2)</div>
            </div>""", unsafe_allow_html=True)
            st.caption("A drop near 0.0 indicates that destroying these components breaks the task behavior.")

        st.markdown("---")
        st.subheader("Pillar Summary")
        pillars_df = pd.DataFrame([
            {"Pillar": "1. Causal Patching", "Metric": "Logit Difference Recovery", "Score": f"{patch_score:.4f}"},
            {"Pillar": "2. Mean Ablation", "Metric": "Performance Loss on Removal", "Score": f"{ablate_score:.4f}"}
        ])
        st.table(pillars_df)
    else:
        st.info("Evaluation has not been run yet. Click 'Evaluate Faithfulness' in the Pipeline tab to compute scores.")


# =============================================================================
# TAB 5: Prompt Playground
# =============================================================================
with tab_inference:
    st.markdown("### 🎮 Interactive Prompt Playground")
    st.markdown("Test how the model processes sentences. For IOI (Indirect Object Identification), the model should predict the indirect object (the name not repeated in the main clause).")

    default_prompt = "When Mary and John went to the store, John gave a drink to"
    user_prompt = st.text_area("Input Prompt", value=default_prompt, height=80)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_name = st.text_input("Expected Answer (IOI Target)", value=" Mary")
    with col_t2:
        distractor_name = st.text_input("Distractor (Subject)", value=" John")

    if st.button("🔮 Predict Next Token", type="primary"):
        with st.spinner("Running forward pass with HookedTransformer..."):
            try:
                model = get_model(target_model)
                tokens = model.to_tokens(user_prompt)
                with torch.no_grad():
                    logits = model(tokens)
                
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                
                # Get Top 5 predictions
                top_k = torch.topk(probs, k=5)
                top_tokens = [model.to_string(t) for t in top_k.indices]
                top_probs = top_k.values.tolist()

                st.subheader("Top Predicted Tokens")
                res_df = pd.DataFrame({
                    "Rank": range(1, 6),
                    "Token": [repr(t) for t in top_tokens],
                    "Probability": [f"{p*100:.2f}%" for p in top_probs]
                })
                st.table(res_df)

                # Target vs Distractor comparison
                target_tok_id = model.to_single_token(target_name)
                distractor_tok_id = model.to_single_token(distractor_name)
                
                target_prob = probs[target_tok_id].item() if target_tok_id is not None else 0.0
                distractor_prob = probs[distractor_tok_id].item() if distractor_tok_id is not None else 0.0

                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    st.metric(f"Prob('{target_name}')", f"{target_prob*100:.2f}%")
                with pcol2:
                    st.metric(f"Prob('{distractor_name}')", f"{distractor_prob*100:.2f}%")
                with pcol3:
                    diff = (target_prob - distractor_prob) * 100
                    st.metric("IOI Margin (Target - Distractor)", f"{diff:+.2f}%")

            except Exception as e:
                st.error(f"Inference error: {e}")


# =============================================================================
# TAB 6: Guide & Code Snippets
# =============================================================================
with tab_docs:
    st.markdown("### 📖 CircuitKIT Quickstart & Python Code")
    st.markdown("You can reproduce everything in this web studio directly via Python or CLI:")

    st.subheader("Python Script")
    st.code(f"""
import circuitkit as ck
from circuitkit import Pipeline

# 1. High-Level Pipeline
pipe = Pipeline("{target_model}", task="{selected_task}")
pipe.discover(algorithm="{algorithm}", sparsity={sparsity})
pipe.evaluate()
pipe.prune()
pipe.export("./checkpoint")

# 2. Granular Functional API
model = ck.load_model("{target_model}")
circuit = ck.discover(model, "{selected_task}", algorithm="{algorithm}", sparsity={sparsity})
ck.export_checkpoint(model, circuit, "./checkpoint")
""", language="python")

    st.subheader("CLI Command")
    st.code(f"""
circuitkit discover --model {target_model} --task {selected_task} --algorithm {algorithm} --sparsity {sparsity}
circuitkit evaluate --model {target_model} --task {selected_task} --circuit circuit_discovery_results.pt
circuitkit prune --model {target_model} --circuit circuit_discovery_results.pt --output ./pruned_checkpoint
""", language="bash")
