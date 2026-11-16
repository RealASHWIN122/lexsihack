#!/usr/bin/env python3
"""
One-Command Publisher for ALL 4 Hackathon Hugging Face Deliverables:
  1. Public HF Model Repo (+ Model Card Article): <username>/tiny-aya-fire-lexsi-ctheta-repair
  2. Public HF Dataset Repo (CuratorKIT Splits + Audit Traces + Benchmarks): <username>/tiny-aya-fire-lexsi-audit-traces
  3. Public HF Gradio Space (Interactive Demo + ROI Calculator): <username>/tiny-aya-fire-lexsi-repair
  4. Public HF Model Collection & Public HF Dataset Collection (required on Page 2 of Hackathon Form)
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, get_token

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
RESULTS_DIR = EXP_DIR / "results"
DATA_DIR = EXP_DIR / "data"


def _resolve_token(cli_token: str | None) -> str:
    if cli_token and cli_token.strip():
        return cli_token.strip()
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "lexsicolab"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    tok = get_token()
    if tok:
        return tok
    try:
        from google.colab import userdata  # type: ignore
        for sname in ("lexsicolab", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            try:
                v = userdata.get(sname)
                if v and v.strip():
                    return v.strip()
            except Exception:
                pass
    except Exception:
        pass
    raise RuntimeError("No Hugging Face token found. Run `hf auth login` or pass `--token hf_...`.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Model, Dataset, Space, and Collections to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="HF Write Token")
    args = parser.parse_args()

    token = _resolve_token(args.token)
    api = HfApi(token=token)
    username = api.whoami()["name"]

    model_repo = f"{username}/tiny-aya-fire-lexsi-ctheta-repair"
    dataset_repo = f"{username}/tiny-aya-fire-lexsi-audit-traces"
    space_repo = f"{username}/tiny-aya-fire-lexsi-repair"

    # 1. Create & Upload Model Repo (+ Model Card Article)
    print(f"1/5 Creating & uploading Model Repo: https://huggingface.co/{model_repo}")
    api.create_repo(repo_id=model_repo, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        article_body = (EXP_DIR / "HF_ARTICLE.md").read_text(encoding="utf-8")
        model_card = (
            "---\n"
            "license: apache-2.0\n"
            "base_model: CohereLabs/tiny-aya-fire\n"
            "language:\n"
            "  - en\n  - hi\n  - bn\n  - ta\n  - te\n  - kn\n"
            "tags:\n"
            "  - lexsi\n  - circuitkit\n  - safetune\n  - auditkit\n  - curatorkit\n  - mechanistic-interpretability\n  - model-repair\n"
            "---\n\n"
            + article_body
        )
        (stage / "README.md").write_text(model_card, encoding="utf-8")
        (stage / "results").mkdir(parents=True, exist_ok=True)
        for fn in (
            "circuit.pt",
            "circuit_nodes.json",
            "circuit_graph.html",
            "safety_units.json",
            "lexsi_provenance.json",
            "alpha_calibration.json",
            "alignment_recipe.yaml",
            "audit_ledger.json",
            "auditkit_report.md",
            "fig1_eap_circuit_attribution.png",
            "fig2_alpha_calibration.png",
            "fig3_triaxial_audit.png",
            "fig4_two_judge_panel.png",
            "fig4b_defensive_10_tasks_heatmap.png",
        ):
            if (RESULTS_DIR / fn).exists():
                shutil.copy2(RESULTS_DIR / fn, stage / fn)
                shutil.copy2(RESULTS_DIR / fn, stage / "results" / fn)
        api.upload_folder(
            folder_path=str(stage),
            repo_id=model_repo,
            repo_type="model",
            commit_message="Upload C-ΔΘ repaired circuit artifacts, provenance, figures, and full technical article",
        )

    # 2. Create & Upload Dataset Repo (CuratorKIT splits + Audit traces + Benchmarks)
    print(f"2/5 Creating & uploading Dataset Repo: https://huggingface.co/datasets/{dataset_repo}")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        ds_card = (
            "---\n"
            "license: apache-2.0\n"
            "task_categories:\n"
            "  - text-generation\n"
            "language:\n"
            "  - en\n  - hi\n  - bn\n  - ta\n  - te\n  - kn\n"
            "tags:\n"
            "  - lexsi\n  - curatorkit\n  - auditkit\n  - safety-benchmarks\n"
            "---\n\n"
            "# Tiny-Aya-Fire Targeted Model Repair — CuratorKIT Splits, Audit Traces & Benchmark Ledger\n\n"
            "Contains the disjoint CuratorKIT contrastive correction datasets (`data.csv`, `sft_messages.jsonl`), "
            "held-out evaluation transcripts (`aya_fire_generations_sample.json`), LLM-as-a-Judge cache (`llm_judge_cache.json`), "
            "and full Triaxial AuditKIT ledger (`audit_ledger.json`, `auditkit_report.md`).\n"
        )
        (stage / "README.md").write_text(ds_card, encoding="utf-8")
        shutil.copy2(DATA_DIR / "data.csv", stage / "correction_pairs_data.csv")
        shutil.copy2(DATA_DIR / "task.yaml", stage / "task.yaml")
        if (RESULTS_DIR / "curatorkit_out").exists():
            shutil.copytree(RESULTS_DIR / "curatorkit_out", stage / "curatorkit_out", dirs_exist_ok=True)
        for fn in (
            "audit_ledger.json",
            "auditkit_report.md",
            "llm_judge_cache.json",
            "aya_fire_generations_sample.json",
            "alpha_calibration.json",
            "lexsi_provenance.json",
        ):
            if (RESULTS_DIR / fn).exists():
                shutil.copy2(RESULTS_DIR / fn, stage / fn)
        api.upload_folder(
            folder_path=str(stage),
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="Upload CuratorKIT curated splits, AuditKIT traces, and benchmark evaluation outputs",
        )

    # 3. Create & Upload Gradio Space
    print(f"3/5 Creating & uploading Gradio Space: https://huggingface.co/spaces/{space_repo}")
    api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(HERE / "app.py", stage / "app.py")
        shutil.copy2(HERE / "README.md", stage / "README.md")
        shutil.copy2(HERE / "requirements.txt", stage / "requirements.txt")
        shutil.copy2(EXP_DIR / "HF_ARTICLE.md", stage / "HF_ARTICLE.md")
        shutil.copytree(HERE / "results", stage / "results", dirs_exist_ok=True)
        api.upload_folder(
            folder_path=str(stage),
            repo_id=space_repo,
            repo_type="space",
            commit_message="Deploy Tiny-Aya-Fire Targeted Model Repair Gradio Space",
        )

    # 4. Create Public Model Collection
    print("4/5 Creating Public Model Collection...")
    model_col = api.create_collection(
        title="Tiny-Aya-Fire Targeted Model Repair (Models & Circuits)",
        description="C-ΔΘ circuit-repaired model artifacts, EAP-IG safety circuits, and base model for CohereLabs/tiny-aya-fire.",
        private=False,
        exists_ok=True,
    )
    for item_id, item_type in [(model_repo, "model"), ("CohereLabs/tiny-aya-fire", "model"), (space_repo, "space")]:
        try:
            api.add_collection_item(
                collection_slug=model_col.slug,
                item_id=item_id,
                item_type=item_type,
                exists_ok=True,
            )
        except Exception:
            pass

    # 5. Create Public Dataset Collection
    print("5/5 Creating Public Dataset Collection...")
    ds_col = api.create_collection(
        title="Tiny-Aya-Fire Targeted Model Repair (Datasets & Traces)",
        description="CuratorKIT disjoint contrastive splits, AuditKIT triaxial benchmark ledgers, and evaluation traces.",
        private=False,
        exists_ok=True,
    )
    try:
        api.add_collection_item(
            collection_slug=ds_col.slug,
            item_id=dataset_repo,
            item_type="dataset",
            exists_ok=True,
        )
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("✓ ALL HUGGING FACE HACKATHON LINKS CREATED & PUBLIC:")
    print("=" * 80)
    print(f"1. Hugging Face model collection *   : {model_col.url}")
    print(f"2. Hugging Face dataset collection * : {ds_col.url}")
    print(f"3. Hugging Face article *            : https://huggingface.co/{model_repo}")
    print(f"   (or GitHub Article URL)           : https://github.com/RealASHWIN122/lexsihack/blob/main/experiments/experiment_02_aya_fire_lexsistack/HF_ARTICLE.md")
    print(f"4. Hugging Face Space (Live Demo)    : https://huggingface.co/spaces/{space_repo}")
    print(f"5. GitHub repo *                     : https://github.com/RealASHWIN122/lexsihack")
    print("=" * 80)


if __name__ == "__main__":
    main()
