#!/usr/bin/env python3
"""
One-Command Public Deployment to Hugging Face Spaces.

Bundles `hf_space/app.py`, `hf_space/README.md`, `hf_space/requirements.txt`, `HF_ARTICLE.md`,
and all lightweight JSON/HTML/PNG results from `results/` (excluding >50MB weight caches)
and pushes directly to a public Gradio Hugging Face Space using your `HF_TOKEN` / `lexsicolab` token.

Usage (Local or Colab):
    python experiments/experiment_02_aya_fire_lexsistack/hf_space/deploy_to_hf_space.py
    python experiments/experiment_02_aya_fire_lexsistack/hf_space/deploy_to_hf_space.py --space-id <username>/tiny-aya-fire-lexsi-repair
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
RESULTS_DIR = EXP_DIR / "results"
BACKUP_DIR = Path("/content/lexsi_checkpoints")


def _resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "lexsicolab"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
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
    raise RuntimeError("No Hugging Face token found. Set HF_TOKEN or pass --token.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Experiment 02 Gradio Space to Hugging Face")
    parser.add_argument(
        "--space-id",
        type=str,
        default=None,
        help="Target Space repo_id (default: <your-hf-username>/tiny-aya-fire-lexsi-repair)",
    )
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (defaults to HF_TOKEN / lexsicolab)")
    args = parser.parse_args()

    token = _resolve_token(args.token)
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    space_id = args.space_id or f"{username}/tiny-aya-fire-lexsi-repair"

    print(f"Creating / updating public Gradio HF Space: https://huggingface.co/spaces/{space_id}")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )

    excluded = {"step3_repairs_cache.pt", "circuit.pt"}
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = Path(tmpdir)
        shutil.copy2(HERE / "app.py", stage / "app.py")
        shutil.copy2(HERE / "README.md", stage / "README.md")
        shutil.copy2(HERE / "requirements.txt", stage / "requirements.txt")
        if (EXP_DIR / "HF_ARTICLE.md").exists():
            shutil.copy2(EXP_DIR / "HF_ARTICLE.md", stage / "HF_ARTICLE.md")

        stage_res = stage / "results"
        stage_res.mkdir(parents=True, exist_ok=True)

        # Copy from RESULTS_DIR and overlay any newer files from BACKUP_DIR
        for src_dir in (RESULTS_DIR, BACKUP_DIR):
            if src_dir.exists():
                for item in src_dir.iterdir():
                    if item.is_file() and item.name not in excluded and not item.name.endswith(".tmp"):
                        shutil.copy2(item, stage_res / item.name)

        api.upload_folder(
            folder_path=str(stage),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy Tiny-Aya-Fire Targeted Model Repair (Lexsi Stack) Space & Results",
        )

    print(f"\n✓ Successfully deployed public Hugging Face Space!")
    print(f"  👉 Live URL: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
