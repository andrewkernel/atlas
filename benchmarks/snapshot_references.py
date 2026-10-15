"""Read-only primary-source metadata capture. Does not execute repository code."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

REPOSITORIES = [
    "pytorch/pytorch", "deepspeedai/DeepSpeed", "huggingface/accelerate",
    "hpcaitech/ColossalAI", "NVIDIA/Megatron-LM", "parasj/checkmate",
    "topal-team/rockmate", "alpa-projects/alpa", "mlflow/mlflow",
    "deep500/deep500", "OrderLab/TrainCheck", "intelland/TrainParity",
    "ScalingIntelligence/KernelBench", "pytorch/examples", "zalandoresearch/fashion-mnist",
]


def gh_json(endpoint: str):
    process = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, timeout=40, check=True)
    return json.loads(process.stdout)


def repository_snapshot(repo: str) -> dict:
    try:
        metadata = gh_json(f"repos/{repo}")
        branch = gh_json(f"repos/{repo}/branches/{metadata['default_branch']}")
        return {"repository": repo, "url": metadata["html_url"], "default_branch": metadata["default_branch"], "head_sha": branch["commit"]["sha"], "archived": metadata["archived"], "pushed_at": metadata["pushed_at"], "license_spdx": (metadata.get("license") or {}).get("spdx_id"), "read_depth": "metadata; detailed capabilities are sourced separately in Markdown"}
    except Exception as exc:
        return {"repository": repo, "error": str(exc)}


if __name__ == "__main__":
    destination = Path("docs/research/competitors/repository-snapshot.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(repository_snapshot, REPOSITORIES))
    payload = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "repositories": results}
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    assets = []
    for kind, name in [("datasets", "Salesforce/wikitext"), ("datasets", "stanfordnlp/sst2"), ("models", "distilbert/distilbert-base-uncased"), ("models", "Qwen/Qwen3-4B"), ("models", "Qwen/Qwen2.5-Coder-7B-Instruct"), ("models", "Qwen/Qwen2.5-Coder-3B-Instruct"), ("models", "microsoft/Phi-4-mini-instruct")]:
        try:
            with urlopen(f"https://huggingface.co/api/{kind}/{name}", timeout=30) as response:
                record = json.load(response)
            assets.append({"kind": kind, "id": name, "revision": record.get("sha"), "license_metadata": (record.get("cardData") or {}).get("license"), "url": f"https://huggingface.co/{'datasets/' if kind == 'datasets' else ''}{name}"})
        except Exception as exc:
            assets.append({"kind": kind, "id": name, "error": str(exc)})
    destination = Path("docs/research/asset-snapshot.json")
    destination.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(), "assets": assets}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assets, indent=2))
