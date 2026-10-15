"""Acquire allowlisted public data/tensor assets; never imports downloaded code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "docs/research/workload-screen/qa-source-screen.json"


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}


def download(url, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if shutil.disk_usage(ROOT).free < expected["bytes"] + 1_000_000_000:
            raise RuntimeError("Keep at least 1 GB free after this asset")
        temporary = target.with_suffix(target.suffix + ".part")
        observed = 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                observed += len(chunk)
                if observed > expected["bytes"]:
                    raise ValueError("Asset exceeds pinned byte count")
                output.write(chunk)
        actual = identity(temporary)
        assert all(actual[key] == value for key, value in expected.items()), target.name
        temporary.replace(target)
    actual = identity(target)
    assert all(actual[key] == value for key, value in expected.items()), target.name
    return {"requested_url": url, **actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-training-assets", action="store_true")
    args = parser.parse_args()
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    model = screen["records"]["bert_model_metadata"]
    assert model["revision"] == "86b5e0934494bd15c9632b12f734a8a67f723594"
    directory = ROOT / "weights/bert-base-uncased-pinned"
    selected = model["selected_files"]
    records = {}
    for item in selected:
        name = item["rfilename"]
        if name == "model.safetensors" and not args.include_training_assets:
            continue
        expected = {"bytes": item["size"]}
        expected["sha256" if "lfs" in item else "git_blob_sha1"] = (
            item["lfs"]["sha256"] if "lfs" in item else item["blobId"])
        url = f"https://huggingface.co/google-bert/bert-base-uncased/resolve/{model['revision']}/{name}"
        records["model/" + name] = download(url, directory / name, expected)
        print("verified", "model/" + name, flush=True)
    if args.include_training_assets:
        url = ("https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
               "eee5fdbf62f8613a7812b03419e6b29617b74fd1/dataset/train-v1.1.json")
        records["data/train-v1.1.json"] = download(url, ROOT / "data/squad/train-v1.1.json",
            {"bytes": 30288272, "git_blob_sha1": "77f1d9a7daee11f7bc293832020f7709f63410ad"})
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_screen_sha256": hashlib.sha256(SCREEN.read_bytes()).hexdigest(),
              "training_assets_requested": args.include_training_assets, "files": records,
              "scope": "Pinned public tensor/data assets only, no foreign scripts, pickle weights or dev/test set. Byte identity is not scientific correctness."}
    output = ROOT / "docs/research/workload-screen/qa-asset-manifest.json"
    output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
