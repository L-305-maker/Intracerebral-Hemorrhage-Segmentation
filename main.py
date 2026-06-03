import argparse
import sys
from pathlib import Path

from src.train_test import main as train_test_main
from src.utils.visualize import visualize


def run_train_test(train_args):
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]] + train_args
    try:
        train_test_main()
    finally:
        sys.argv = old_argv


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


def split_mode_args(argv):
    if "--mode" not in argv:
        raise SystemExit("Please provide --mode train_test, --mode test or --mode visualize")
    mode_index = argv.index("--mode")
    if mode_index + 1 >= len(argv):
        raise SystemExit("Please provide a value for --mode")
    mode = argv[mode_index + 1]
    rest_args = argv[:mode_index] + argv[mode_index + 2:]
    return mode, rest_args


def main():
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parse_main_args()
        return

    mode, extra_args = split_mode_args(sys.argv[1:])

    if mode == "train_test":
        run_train_test(["--mode", "train", *extra_args])

    elif mode == "test":
        run_train_test(["--mode", "eval", *extra_args])

    elif mode == "visualize":
        args, _ = parse_main_args()
        visualize(
            config_dir=Path(args.config_dir),
            out_dir=Path(args.out_dir),
        )
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
