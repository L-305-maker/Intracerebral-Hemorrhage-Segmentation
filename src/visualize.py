import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPARE_KEYS = [
    "model",
    "criterion",
    "learning_rate",
    "image_size",
    "base_channels",
    "batch_size",
    "dropout",
    "BatchNorm",
    "Double_Conv",
]
METRICS = ["valid_Dice", "valid_IoU", "test_Dice", "test_IoU"]


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", str(text)).strip("_")


def load_records(config_dir: Path):
    records = []
    for path in sorted(config_dir.glob("*.json")):
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        params = data.get("parameters", {})
        train = data.get("results", {}).get("train") or {}
        test = data.get("results", {}).get("test") or {}
        best_valid = train.get("best_valid") or {}
        records.append({
            "name": data.get("experiment_name", path.stem),
            "params": params,
            "metrics": {
                "valid_Dice": best_valid.get("Dice"),
                "valid_IoU": best_valid.get("IoU"),
                "test_Dice": test.get("Dice"),
                "test_IoU": test.get("IoU"),
            },
        })
    return records


def plot_group(records, title: str, out_path: Path, label_key=None):
    labels = [str(label_key(item) if label_key else item["name"]) for item in records]
    x = list(range(len(records)))
    width = 0.18

    plt.figure(figsize=(max(8, len(records) * 1.2), 5))
    for idx, metric in enumerate(METRICS):
        values = [item["metrics"].get(metric) for item in records]
        values = [0 if value is None else value for value in values]
        xs = [pos + (idx - 1.5) * width for pos in x]
        plt.bar(xs, values, width=width, label=metric)

    plt.title(title)
    plt.ylabel("score")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.subplots_adjust(bottom=0.35, top=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def find_single_variable_groups(records):
    groups = []
    for key in COMPARE_KEYS:
        buckets = defaultdict(list)
        for item in records:
            params = item["params"]
            signature = tuple((k, params.get(k)) for k in COMPARE_KEYS if k != key)
            buckets[signature].append(item)
        for items in buckets.values():
            values = {item["params"].get(key) for item in items}
            if len(items) >= 2 and len(values) >= 2:
                items.sort(key=lambda item: str(item["params"].get(key)))
                groups.append((key, items))
    return groups


def visualize(config_dir: Path, out_dir: Path):
    records = load_records(config_dir)
    if not records:
        print(f"No experiment json files found in {config_dir}")
        return []

    saved = []
    all_path = out_dir / "all_experiments.png"
    plot_group(records, "All Experiments", all_path)
    saved.append(all_path)

    for idx, (key, items) in enumerate(find_single_variable_groups(records), start=1):
        title = f"Compare {key}"
        out_path = out_dir / f"compare_{idx:02d}_{safe_name(key)}.png"
        plot_group(items, title, out_path, label_key=lambda item: item["params"].get(key))
        saved.append(out_path)

    for path in saved:
        print(f"Saved {path}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment records from configs/")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--out_dir", type=str, default="outputs/visualizations")
    args = parser.parse_args()
    visualize(Path(args.config_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
