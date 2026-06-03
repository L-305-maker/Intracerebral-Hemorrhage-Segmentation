import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["train_Dice", "train_IoU", "valid_Dice", "valid_IoU"]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json_records(config_dir, group_name=None):
    records = []
    config_dir = Path(config_dir)
    group_name = group_name or config_dir.name

    if not config_dir.exists():
        print(f"[Warning] config_dir not found: {config_dir}")
        return records

    for json_path in sorted(config_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning] skip {json_path}: {e}")
            continue

        results = data.get("results", {})
        train_result = results.get("train") or {}
        best_valid = train_result.get("best_valid") or {}
        history = train_result.get("history") or []

        train_metrics = get_train_metrics(history, best_valid.get("epoch"))

        records.append(
            {
                "name": data.get("experiment_name", json_path.stem),
                "group": group_name,
                "metrics": {
                    "train_Dice": train_metrics.get("Dice"),
                    "train_IoU": train_metrics.get("IoU"),
                    "valid_Dice": safe_float(best_valid.get("Dice")),
                    "valid_IoU": safe_float(best_valid.get("IoU")),
                },
            }
        )

    return records


def get_train_metrics(history, best_epoch):
    if not history:
        return {"Dice": None, "IoU": None}

    selected = history[-1]
    for item in history:
        if item.get("epoch") == best_epoch:
            selected = item
            break

    train_metrics = selected.get("train") or {}
    return {
        "Dice": safe_float(train_metrics.get("Dice")),
        "IoU": safe_float(train_metrics.get("IoU")),
    }


def get_best_record(records):
    if not records:
        return None

    return max(
        records,
        key=lambda record: record["metrics"].get("valid_Dice") or -1,
    )


def plot_bar(records, out_path, title):
    if not records:
        print(f"[Warning] no records for {title}")
        return None

    x = list(range(len(records)))
    width = 0.8 / len(METRICS)
    labels = [
        f"{record['group']}_{i + 1}"
        if len({item["group"] for item in records}) > 1
        else f"exp_{i + 1}"
        for i, record in enumerate(records)
    ]

    plt.figure(figsize=(max(8, len(records) * 1.2), 5))

    for metric_index, metric in enumerate(METRICS):
        values = [
            record["metrics"].get(metric)
            if record["metrics"].get(metric) is not None
            else 0
            for record in records
        ]
        positions = [
            pos + (metric_index - len(METRICS) / 2) * width + width / 2
            for pos in x
        ]
        plt.bar(positions, values, width=width, label=metric)

    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"Saved: {out_path}")
    return out_path


def visualize(config_dir="configs", out_dir="outputs/visualizations"):
    records = read_json_records(config_dir)
    out_dir = Path(out_dir)

    return [
        plot_bar(
            records=records,
            out_path=out_dir / "all_experiments.png",
            title="All Experiments",
        )
    ]


def compare_dirs(dir_a, dir_b, out_dir="outputs/visualizations", label_a=None, label_b=None):
    records_a = read_json_records(dir_a, label_a or Path(dir_a).name)
    records_b = read_json_records(dir_b, label_b or Path(dir_b).name)
    out_dir = Path(out_dir)

    best_records = [
        record
        for record in [get_best_record(records_a), get_best_record(records_b)]
        if record is not None
    ]
    all_records = records_a + records_b

    return [
        plot_bar(
            records=best_records,
            out_path=out_dir / "two_dirs_best.png",
            title="Best Experiment of Two Folders",
        ),
        plot_bar(
            records=all_records,
            out_path=out_dir / "two_dirs_all.png",
            title="All Experiments of Two Folders",
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "compare_dirs"], default="all")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--dir_a", type=str)
    parser.add_argument("--dir_b", type=str)
    parser.add_argument("--label_a", type=str)
    parser.add_argument("--label_b", type=str)
    parser.add_argument("--out_dir", type=str, default="outputs/visualizations")

    args = parser.parse_args()

    if args.mode == "compare_dirs":
        if args.dir_a is None or args.dir_b is None:
            raise SystemExit("--dir_a and --dir_b are required")
        compare_dirs(
            dir_a=args.dir_a,
            dir_b=args.dir_b,
            out_dir=args.out_dir,
            label_a=args.label_a,
            label_b=args.label_b,
        )
    else:
        visualize(config_dir=args.config_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
