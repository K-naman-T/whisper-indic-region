#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Whisper on prepared IndicVoices manifests")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_cfg = load_yaml(PROJECT_ROOT / args.data_config)
    model_cfg = load_yaml(PROJECT_ROOT / args.model_config)
    train_cfg = load_yaml(PROJECT_ROOT / args.train_config)
    paths_cfg = load_yaml(PROJECT_ROOT / args.paths_config)

    manifests_dir = Path(paths_cfg["data_dir"])
    data_files = {
        "train": str(manifests_dir / "train.parquet"),
        "validation": str(manifests_dir / "dev.parquet"),
    }
    ds = load_dataset("parquet", data_files=data_files)
    ds = ds.cast_column(data_cfg["audio_column"], Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(model_cfg["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["model_name"])
    model.config.use_cache = False

    audio_column = data_cfg["audio_column"]
    text_column = data_cfg["text_column"]
    language_tokens = model_cfg["language_tokens"]

    def prepare_example(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[audio_column]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        lang = batch["language"]
        whisper_lang = language_tokens.get(lang, "hi")
        processor.tokenizer.set_prefix_tokens(language=whisper_lang, task=model_cfg["task"])
        batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
        return batch

    ds = ds.map(prepare_example, remove_columns=ds["train"].column_names)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    t = train_cfg["train"]
    e = train_cfg["eval"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=e["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        max_steps=t["max_steps"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        fp16=train_cfg["precision"] == "fp16",
        bf16=train_cfg["precision"] == "bf16",
        evaluation_strategy="steps",
        eval_steps=e["eval_steps"],
        save_steps=e["save_steps"],
        logging_steps=e["logging_steps"],
        save_total_limit=e["save_total_limit"],
        predict_with_generate=True,
        generation_max_length=model_cfg["generation"]["max_new_tokens"],
        load_best_model_at_end=True,
        metric_for_best_model=e["metric_for_best_model"],
        greater_is_better=e["greater_is_better"],
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    metrics = trainer.evaluate()
    report_path = Path(paths_cfg["reports_dir"]) / "train_eval_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved training metrics to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
