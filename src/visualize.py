import argparse
import csv
import json
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
    "Attention",
    "BatchNorm",
    "Double_Conv",
]

METRICS = ["valid_Dice", "valid_IoU", "test_Dice", "test_IoU"]


def get_value(data, keys, default=None):
    """
    从嵌套字典中安全取值。
    例如：
    get_value(data, ["results", "train", "best_valid", "Dice"])
    """
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key)
    return default if data is None else data


def to_float(value):
    """
    将指标转成 float。
    转换失败时返回 None，避免把缺失指标错误当成 0。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_records(config_dir):
    """
    读取 configs/ 目录下的实验 JSON 文件。
    """
    config_dir = Path(config_dir)
    records = []

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

        params = data.get("parameters", {})
        results = data.get("results", {})

        metrics = {
            "valid_Dice": to_float(
                get_value(results, ["train", "best_valid", "Dice"])
            ),
            "valid_IoU": to_float(
                get_value(results, ["train", "best_valid", "IoU"])
            ),
            "test_Dice": to_float(
                get_value(results, ["test", "Dice"])
            ),
            "test_IoU": to_float(
                get_value(results, ["test", "IoU"])
            ),
        }

        records.append(
            {
                "name": data.get("experiment_name", json_path.stem),
                "created_at": data.get("created_at", ""),
                "path": str(json_path),
                "params": params,
                "metrics": metrics,
            }
        )

    return records


def save_summary(records, out_dir):
    """
    保存实验汇总表。
    图中使用 exp_001 这种短编号，具体参数和指标放到 CSV 中。
    """
    out_path = Path(out_dir) / "records_summary.csv"

    fieldnames = [
        "exp_id",
        "experiment_name",
        "created_at",
        "json_path",
        *COMPARE_KEYS,
        *METRICS,
    ]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, record in enumerate(records, start=1):
            row = {
                "exp_id": f"exp_{i:03d}",
                "experiment_name": record["name"],
                "created_at": record["created_at"],
                "json_path": record["path"],
            }

            for key in COMPARE_KEYS:
                row[key] = record["params"].get(key, "NA")

            for metric in METRICS:
                value = record["metrics"].get(metric)
                row[metric] = "NA" if value is None else round(value, 6)

            writer.writerow(row)

    return out_path


def plot_records(records, title, out_path, labels):
    """
    绘制实验指标对比图。
    """
    if not records:
        return

    available_metrics = [
        metric
        for metric in METRICS
        if any(record["metrics"].get(metric) is not None for record in records)
    ]

    if not available_metrics:
        print(f"[Warning] no available metrics for {title}")
        return

    x = list(range(len(records)))
    width = 0.8 / len(available_metrics)

    plt.figure(figsize=(max(8, len(records) * 1.3), 5))

    for i, metric in enumerate(available_metrics):
        values = [
            record["metrics"].get(metric, float("nan"))
            for record in records
        ]

        positions = [
            pos + (i - len(available_metrics) / 2) * width + width / 2
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

    plt.savefig(out_path, dpi=160)
    plt.close()


def find_single_variable_groups(records):
    """
    自动寻找单变量实验组。

    例如：
    除 learning_rate 不同外，其他参数都相同，
    就认为这是一组 learning_rate 消融实验。
    """
    groups = []

    for target_key in COMPARE_KEYS:
        buckets = defaultdict(list)

        for record in records:
            params = record["params"]

            signature = tuple(
                (key, params.get(key))
                for key in COMPARE_KEYS
                if key != target_key
            )

            buckets[signature].append(record)

        for group_records in buckets.values():
            values = {
                record["params"].get(target_key)
                for record in group_records
            }

            if len(group_records) >= 2 and len(values) >= 2:
                groups.append((target_key, group_records))

    return groups


def visualize(config_dir="configs", out_dir="outputs/visualizations"):
    """
    可视化统一入口。
    main.py 可以直接调用这个函数。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(config_dir)

    if not records:
        print("No valid experiment records found.")
        return []

    saved_files = []

    summary_path = save_summary(records, out_dir)
    saved_files.append(summary_path)

    all_labels = [f"exp_{i:03d}" for i in range(1, len(records) + 1)]
    all_path = out_dir / "all_experiments.png"

    plot_records(
        records=records,
        title="All Experiments",
        out_path=all_path,
        labels=all_labels,
    )
    saved_files.append(all_path)

    groups = find_single_variable_groups(records)

    for i, (key, group_records) in enumerate(groups, start=1):
        labels = [
            str(record["params"].get(key))
            for record in group_records
        ]

        out_path = out_dir / f"compare_{i:02d}_{key}.png"

        plot_records(
            records=group_records,
            title=f"Compare {key}",
            out_path=out_path,
            labels=labels,
        )

        saved_files.append(out_path)

    print("Visualization finished:")
    for path in saved_files:
        print(f"- {path}")

    return saved_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--out_dir", type=str, default="outputs/visualizations")

    args = parser.parse_args()

    visualize(
        config_dir=args.config_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()