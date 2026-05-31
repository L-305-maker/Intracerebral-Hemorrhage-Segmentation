import json
from datetime import datetime
from pathlib import Path


def _safe_name(value) -> str:
    text = str(value).strip().replace(" ", "_").replace("+", "plus")
    keep_chars = []
    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            keep_chars.append(char)
        else:
            keep_chars.append("_")
    return "".join(keep_chars)


def _metrics_to_float(metrics):
    if metrics is None:
        return None
    result = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            result[key] = _metrics_to_float(value)
        elif isinstance(value, list):
            result[key] = [_metrics_to_float(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, (int, float, str)) or value is None:
            result[key] = value
        else:
            result[key] = float(value)
    return result


def build_experiment_name(args) -> str:
    parts = [
        args.model,
        args.criterion,
        f"img{args.image_size}",
        f"bs{args.batch_size}",
        f"lr{args.lr}",
        f"base{args.base_channels}",
    ]
    if args.model in {"Advanced_model", "top_Unet"}:
        if args.Double_Conv:
            parts.append("DoubleConv")
        if getattr(args, "Residual", False):
            parts.append("Residual")
        if args.BatchNorm:
            parts.append("BatchNorm")
        if args.Attention:
            parts.append("Attention")
        if getattr(args, "ASPP", False):
            parts.append("ASPP")
        if args.dropout > 0:
            parts.append(f"dropout{args.dropout}")
    return _safe_name("_".join(parts))


def collect_model_parameters(args, model) -> dict:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    return {
        "model": args.model,
        "criterion": args.criterion,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "data_dir": args.data_dir,
        "num_workers": args.num_workers,
        "save_dir": args.save_dir,
        "dropout": args.dropout,
        "Attention": args.Attention,
        "BatchNorm": args.BatchNorm,
        "Double_Conv": args.Double_Conv,
        "Residual": getattr(args, "Residual", False),
        "ASPP": getattr(args, "ASPP", False),
        "mode": args.mode,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def save_experiment_record(args, model, train_result=None, test_result=None) -> Path:
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = build_experiment_name(args)
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = config_dir / f"{experiment_name}_{created_at}.json"

    record = {
        "experiment_name": experiment_name,
        "created_at": created_at,
        "parameters": collect_model_parameters(args, model),
        "results": {
            "train": _metrics_to_float(train_result),
            "test": _metrics_to_float(test_result),
        },
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)

    return config_path
