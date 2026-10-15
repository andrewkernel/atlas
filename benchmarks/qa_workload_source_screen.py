"""Read-only provenance screen for a possible BERT/SQuAD research workload."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BERT_SOURCE = "eedf5716ce1268e56f0a50264a88cafad334ac61"
SQUAD_SOURCE = "eee5fdbf62f8613a7812b03419e6b29617b74fd1"
URLS = {
    "bert_readme": f"https://raw.githubusercontent.com/google-research/bert/{BERT_SOURCE}/README.md",
    "bert_run_squad": f"https://raw.githubusercontent.com/google-research/bert/{BERT_SOURCE}/run_squad.py",
    "squad_readme": f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SQUAD_SOURCE}/README.md",
    "squad_home": "https://rajpurkar.github.io/SQuAD-explorer/",
    "bert_model_metadata": "https://huggingface.co/api/models/google-bert/bert-base-uncased?blobs=true",
}


def main():
    cache = ROOT / ".tools/qa-source-screen"
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in URLS.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
                resolved = response.url
            (cache / name).write_bytes(data)
            row = {"url": url, "resolved_url": resolved, "bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest(), "status": "retrieved"}
            if name == "bert_model_metadata":
                metadata = json.loads(data)
                row["revision"] = metadata["sha"]
                selected = {"model.safetensors", "LICENSE", "README.md", "config.json",
                            "tokenizer.json", "tokenizer_config.json", "vocab.txt"}
                row["selected_files"] = [item for item in metadata["siblings"]
                                         if item["rfilename"] in selected]
        except Exception as error:
            row = {"url": url, "status": "error", "error_type": type(error).__name__,
                   "message": str(error)}
        records[name] = row
        print(name, row["status"], flush=True)
    output = ROOT / "docs/research/workload-screen/qa-source-screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bert_source_revision": BERT_SOURCE, "squad_source_revision": SQUAD_SOURCE,
              "records": records,
              "scope": "Suitability/source screen only; downloaded source is read as text, not run. No model weights, dataset training, OOM or fit result."}
    output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
