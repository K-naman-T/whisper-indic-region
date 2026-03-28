from __future__ import annotations

from collections import defaultdict

import evaluate


def compute_grouped_wer_cer(predictions: list[str], references: list[str], languages: list[str]) -> dict[str, dict[str, float]]:
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    grouped_pred: dict[str, list[str]] = defaultdict(list)
    grouped_ref: dict[str, list[str]] = defaultdict(list)

    for pred, ref, lang in zip(predictions, references, languages):
        grouped_pred[lang].append(pred)
        grouped_ref[lang].append(ref)

    results: dict[str, dict[str, float]] = {}
    for lang in sorted(grouped_ref):
        refs = grouped_ref[lang]
        preds = grouped_pred[lang]
        results[lang] = {
            "wer": float(wer_metric.compute(predictions=preds, references=refs)),
            "cer": float(cer_metric.compute(predictions=preds, references=refs)),
            "count": float(len(refs)),
        }
    return results
