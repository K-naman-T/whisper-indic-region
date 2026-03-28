#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import compute_grouped_wer_cer
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Whisper model on test manifest")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_cfg = load_yaml(PROJECT_ROOT / args.data_config)
    model_cfg = load_yaml(PROJECT_ROOT / args.model_config)
    paths_cfg = load_yaml(PROJECT_ROOT / args.paths_config)

    test_manifest = Path(paths_cfg["data_dir"]) / "test.parquet"
    ds = load_dataset("parquet", data_files={"test": str(test_manifest)})["test"]
    ds = ds.cast_column(data_cfg["audio_column"], Audio(sampling_rate=16000))
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    processor = WhisperProcessor.from_pretrained(args.checkpoint_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions: list[str] = []
    references: list[str] = []
    languages: list[str] = []

    for row in tqdm(ds, desc="Evaluating"):
        audio = row[data_cfg["audio_column"]]
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features.to(device)

        whisper_lang = model_cfg["language_tokens"].get(row["language"], "hi")
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task=model_cfg["task"])
        pred_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]

        predictions.append(pred_text)
        references.append(str(row[data_cfg["text_column"]]))
        languages.append(str(row["language"]))

    grouped = compute_grouped_wer_cer(predictions, references, languages)

    report = {
        "checkpoint": args.checkpoint_dir,
        "num_samples": len(predictions),
        "metrics_by_language": grouped,
    }
    report_path = Path(paths_cfg["reports_dir"]) / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved evaluation report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
