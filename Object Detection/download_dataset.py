"""[OPTIONAL] Download a YOLO-format dataset from Roboflow.

This is **only needed for fine-tuning or evaluation**. Production inference
uses pretrained `yolo11n.pt` and never touches Roboflow.

Reads credentials from .env (or env vars):
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION

Usage:
    python download_dataset.py                 # uses .env defaults
    python download_dataset.py --version 50    # override version
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE", "microsoft"))
    parser.add_argument("--project", default=os.environ.get("ROBOFLOW_PROJECT", "coco"))
    parser.add_argument("--version", type=int,
                        default=int(os.environ.get("ROBOFLOW_VERSION", "50")))
    parser.add_argument("--format", default="yolov8",
                        help="Roboflow export format. yolov8 works for YOLO11 too.")
    parser.add_argument("--out", default="./datasets")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY missing (set it in .env or as an env var)")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format)

    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"\nDataset ready at: {dataset.location}")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
