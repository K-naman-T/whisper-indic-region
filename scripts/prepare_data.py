#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import create_speaker_disjoint_split, load_and_prepare, write_manifests
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IndicVoices subset for Whisper training")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_cfg = load_yaml(PROJECT_ROOT / args.data_config)
    paths_cfg = load_yaml(PROJECT_ROOT / args.paths_config)

    dataset = load_and_prepare(data_cfg, cache_dir=paths_cfg["cache_dir"])
    train_pool = dataset["train"]
    valid_holdout = dataset["valid"]

    split_cfg = data_cfg["split_strategy"]
    train_final, dev_final = create_speaker_disjoint_split(
        train_pool,
        speaker_column=data_cfg["speaker_column"],
        dev_fraction=split_cfg["internal_dev_fraction"],
        seed=split_cfg["random_seed"],
    )

    write_manifests(train_final, dev_final, valid_holdout, out_dir=paths_cfg["data_dir"])

    summary = {
        "train_samples": len(train_final),
        "dev_samples": len(dev_final),
        "test_samples": len(valid_holdout),
        "languages": data_cfg["languages"],
    }
    report_path = Path(paths_cfg["reports_dir"]) / "data_prep_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Data preparation complete")
    print(json.dumps(summary, indent=2))
    print(f"Saved manifests to {paths_cfg['data_dir']}")
    print(f"Saved summary report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
