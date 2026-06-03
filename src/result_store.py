import json
from datetime import datetime
from pathlib import Path


def safe_name(text):
    text = str(text).replace("+", "plus").replace(" ", "_")
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in text)


def build_experiment_name(args):
    parts = [
        args.model,
        args.criterion,
        f"img{args.image_size}",
        f"bs{args.batch_size}",
        f"lr{args.lr}",
        f"base{args.base_channels}",
    ]

    for name in ["BatchNorm", "Double_Conv", "Residual", "Attention", "ASPP", "augment", "lr_scheduler"]:
        if getattr(args, name, False):
            parts.append(name)

    if getattr(args, "roi_crop", False):
        parts.append(f"roiT{args.roi_threshold}_P{args.roi_padding}")

    if args.dropout > 0:
        parts.append(f"dropout{args.dropout}")

    return safe_name("_".join(parts))


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return float(value)


def save_experiment_record(args, model, train_result=None, test_result=None):
    experiment_name = build_experiment_name(args)
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.config_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment_name": experiment_name,
        "created_at": created_at,
        "parameters": vars(args),
        "model_parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "results": {
            "train": to_jsonable(train_result),
            "test": to_jsonable(test_result),
        },
    }

    config_path = experiment_dir / f"{created_at}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return config_path
