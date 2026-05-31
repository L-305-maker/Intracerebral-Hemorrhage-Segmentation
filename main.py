import argparse
import sys
from pathlib import Path

from src.train_test import main as train_test_main
from src.visualize import visualize


def run_train_test(train_args):
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]] + train_args
    try:
        train_test_main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description="统一入口：训练/测试或实验结果可视化")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train_test", help="运行训练、验证、测试或评估")
    train_parser.add_argument("train_args", nargs=argparse.REMAINDER)

    visualize_parser = subparsers.add_parser("visualize", help="绘制实验结果对比图")
    visualize_parser.add_argument("--config_dir", type=str, default="configs")
    visualize_parser.add_argument("--out_dir", type=str, default="outputs/visualizations")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if sys.argv[1] in {"-h", "--help"}:
        parser.parse_args()
        return

    if sys.argv[1] == "train_test":
        run_train_test(sys.argv[2:])
        return

    if sys.argv[1].startswith("-"):
        run_train_test(sys.argv[1:])
        return

    args = parser.parse_args()
    if args.command == "train_test":
        run_train_test(args.train_args)
    elif args.command == "visualize":
        visualize(Path(args.config_dir), Path(args.out_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
