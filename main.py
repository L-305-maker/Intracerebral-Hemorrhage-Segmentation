import argparse
from pathlib import Path

from src.train_test import main as train_test_main
from src.visualize import visualize


def parse_main_args():
    parser = argparse.ArgumentParser(
        description="Unified entry for train, test and visualize"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train_test", "test", "visualize"],
        help=(
            "运行模式："
            "train_test 表示训练+验证+测试；"
            "test 表示只测试已有模型；"
            "visualize 表示实验结果可视化。"
        ),
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="visualize 模式下，实验 JSON 文件所在目录",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/visualizations",
        help="visualize 模式下，可视化结果输出目录",
    )

    args, extra_args = parser.parse_known_args()
    return args, extra_args


def main():
    args, extra_args = parse_main_args()

    if args.mode == "train_test":
        train_test_main(["--mode", "train", *extra_args])

    elif args.mode == "test":
        train_test_main(["--mode", "eval", *extra_args])

    elif args.mode == "visualize":
        visualize(
            config_dir=Path(args.config_dir),
            out_dir=Path(args.out_dir),
        )


if __name__ == "__main__":
    main()