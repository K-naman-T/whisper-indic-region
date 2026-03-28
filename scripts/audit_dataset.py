#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit IndicVoices language subsets")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_cfg = load_yaml(PROJECT_ROOT / args.data_config)
    paths_cfg = load_yaml(PROJECT_ROOT / args.paths_config)

    summary: dict[str, dict[str, float | int]] = {}
    for lang in data_cfg["languages"]:
        train = load_dataset(data_cfg["dataset_name"], lang, split="train", cache_dir=paths_cfg["cache_dir"])
        valid = load_dataset(data_cfg["dataset_name"], lang, split="valid", cache_dir=paths_cfg["cache_dir"])

        train_duration = float(sum(train["duration"]))
        valid_duration = float(sum(valid["duration"]))
        summary[lang] = {
            "train_samples": len(train),
            "valid_samples": len(valid),
            "train_hours": round(train_duration / 3600.0, 2),
            "valid_hours": round(valid_duration / 3600.0, 2),
            "unique_speakers_train": len(set(train[data_cfg["speaker_column"]])),
        }

    report_path = Path(paths_cfg["reports_dir"]) / "dataset_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved audit report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
