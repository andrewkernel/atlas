"""Original explicit QA feature builder; local tokenizer data, no remote code."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"questions": 512, "question_seed": 2401, "feature_seed": 2402,
          "sequence_length": 384, "maximum_question_tokens": 64,
          "context_window_advance_tokens": 128}


def window_starts(length, capacity, advance):
    if length <= 0 or capacity <= 0 or not 0 < advance <= capacity:
        raise ValueError("Positive context/capacity and gap-free advance required")
    result = []
    start = 0
    while True:
        result.append(start)
        if start + capacity >= length:
            return result
        start += advance


def answer_tokens(offsets, start, end):
    if not 0 <= start < end:
        raise ValueError("Answer character span must be nonempty")
    first = next((i for i, (a, b) in enumerate(offsets) if a <= start < b), None)
    last = next((i for i, (a, b) in reversed(list(enumerate(offsets))) if a < end <= b), None)
    if first is None or last is None or first > last:
        raise ValueError("Answer endpoint does not map into tokenizer offsets")
    return first, last


def window_positions(answer_first, answer_last, window_start, window_length, question_length):
    if window_start <= answer_first <= answer_last < window_start + window_length:
        prefix = question_length + 2
        return prefix + answer_first - window_start, prefix + answer_last - window_start
    return 0, 0


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def main():
    import numpy as np
    from tokenizers import Tokenizer

    assets_path = ROOT / "docs/research/workload-screen/qa-asset-manifest.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    data_path = ROOT / "data/squad/train-v1.1.json"
    tokenizer_path = ROOT / "weights/bert-base-uncased-pinned/tokenizer.json"
    for path, key in [(data_path, "data/train-v1.1.json"), (tokenizer_path, "model/tokenizer.json")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == assets["files"][key]["sha256"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    assert tokenizer.token_to_id("[CLS]") == 101 and tokenizer.token_to_id("[SEP]") == 102
    assert tokenizer.token_to_id("[PAD]") == 0
    # Original Unicode offset control: answer follows a multibyte character.
    text = "A café has a blue door."
    begin = text.index("blue")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    first, last = answer_tokens(encoded.offsets, begin, begin + len("blue"))
    assert text[encoded.offsets[first][0]:encoded.offsets[last][1]] == "blue"
    source = json.loads(data_path.read_text(encoding="utf-8"))
    assert str(source["version"]) == "1.1"
    questions = []
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                questions.append({"id": qa["id"], "question": qa["question"],
                                  "answers": qa["answers"], "context": paragraph["context"],
                                  "article": article["title"]})
    assert len({qa["id"] for qa in questions}) == len(questions)
    selected = random.Random(CONFIG["question_seed"]).sample(questions, CONFIG["questions"])
    rows = []
    counts = []
    truncated_queries = 0
    for qa in selected:
        question = tokenizer.encode(qa["question"].lstrip(), add_special_tokens=False).ids
        truncated_queries += len(question) > CONFIG["maximum_question_tokens"]
        question = question[:CONFIG["maximum_question_tokens"]]
        context = tokenizer.encode(qa["context"], add_special_tokens=False)
        answer = qa["answers"][0]
        start = answer["answer_start"]
        end = start + len(answer["text"])
        assert qa["context"][start:end] == answer["text"]
        answer_first, answer_last = answer_tokens(context.offsets, start, end)
        capacity = CONFIG["sequence_length"] - len(question) - 3
        starts = window_starts(len(context.ids), capacity, CONFIG["context_window_advance_tokens"])
        covered = 0
        for offset in starts:
            chunk = context.ids[offset:offset + capacity]
            ids = [101] + question + [102] + chunk + [102]
            types = [0] * (len(question) + 2) + [1] * (len(chunk) + 1)
            mask = [1] * len(ids)
            padding = CONFIG["sequence_length"] - len(ids)
            ids += [0] * padding
            types += [0] * padding
            mask += [0] * padding
            a, b = window_positions(answer_first, answer_last, offset, len(chunk), len(question))
            if a:
                covered += 1
                assert types[a] == types[b] == 1 and mask[a] == mask[b] == 1
                assert context.offsets[offset + a - len(question) - 2][0] <= start
                assert context.offsets[offset + b - len(question) - 2][1] >= end
            assert 0 <= a <= b < CONFIG["sequence_length"]
            rows.append({"input_ids": ids, "attention_mask": mask, "token_type_ids": types,
                         "start_positions": a, "end_positions": b, "question_id": qa["id"]})
        assert covered > 0, "No answer-containing window in a selected question"
        counts.append({"id_sha256": hashlib.sha256(qa["id"].encode()).hexdigest(),
                       "features": len(starts), "answer_containing_features": covered})
    random.Random(CONFIG["feature_seed"]).shuffle(rows)
    names = ("input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions")
    arrays = {name: np.asarray([row[name] for row in rows], dtype="<i8") for name in names}
    output = ROOT / "data/squad/qa-resource-features.npz"
    np.savez(output, **arrays)
    details = {name: {"shape": list(a.shape), "dtype": str(a.dtype),
                      "sha256_c_order_little_endian": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
               for name, a in arrays.items()}
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "asset_manifest_canonical_sha256": canonical(assets), "config": CONFIG,
              "source_questions": len(questions), "selected_questions": len(selected),
              "selected_articles": len(Counter(qa["article"] for qa in selected)),
              "features": len(rows), "question_counts": counts, "truncated_queries": truncated_queries,
              "cls_target_features": sum(row["start_positions"] == 0 for row in rows),
              "feature_question_order_digest": canonical([row["question_id"] for row in rows]),
              "arrays": details, "npz_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "unicode_offset_control": "passed",
              "scope": "Authored resource-study feature subset, not complete SQuAD training/evaluation or exact Google preprocessing replication. Advance means window-start increment, not overlap. No dev/test data."}
    target = ROOT / "docs/research/workload-screen/qa-feature-manifest.json"
    target.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({key: result[key] for key in ("source_questions", "selected_questions", "selected_articles", "features", "truncated_queries", "cls_target_features", "unicode_offset_control")}))


if __name__ == "__main__":
    main()
