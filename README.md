# Whisper Indic Region Training Repo

Train and evaluate `openai/whisper-large-v3` for region-focused ASR on IndicVoices using:

- Hindi
- Maithili
- Bengali
- Urdu

This project is set up for an A6000-first workflow and can be cloned to a GPU server.

## What is included

- Non-blocking doctor script (`scripts/doctor.py`) that runs all checks and writes JSON report
- Dataset audit and preparation scripts
- Starter training script for Whisper fine-tuning
- Evaluation script with per-language WER/CER output
- YAML configs for data/model/train/path settings

## Repository layout

```text
configs/
scripts/
src/
outputs/
requirements.txt
.env.example
```

## 1) Local/Server setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

Set your Hugging Face token in `.env` and export it:

```bash
set -a
source .env
set +a
```

## 2) Run doctor checks

```bash
python scripts/doctor.py
```

The doctor script always runs the full checklist, even when checks fail.

- Terminal summary: `PASS | WARN | FAIL` per check
- JSON report: `outputs/reports/doctor_report.json`

## 3) Audit dataset sizes

```bash
python scripts/audit_dataset.py
```

Output report: `outputs/reports/dataset_audit.json`

## 4) Prepare data manifests

```bash
python scripts/prepare_data.py
```

Produces:

- `outputs/manifests/train.parquet`
- `outputs/manifests/dev.parquet`
- `outputs/manifests/test.parquet`
- `outputs/reports/data_prep_summary.json`

Notes:

- Uses `normalized` text column
- Applies duration/text filters from `configs/data.yaml`
- Applies numeral normalization to `0-9`
- Creates internal speaker-disjoint dev split from train

## 5) Train model

```bash
python scripts/train.py
```

Key outputs:

- Checkpoints under `outputs/checkpoints/`
- Training metrics report `outputs/reports/train_eval_metrics.json`

## 6) Evaluate model

```bash
python scripts/evaluate.py --checkpoint-dir outputs/checkpoints
```

Output report:

- `outputs/reports/evaluation_report.json`

## Config files

- `configs/data.yaml`: languages, filtering, split strategy, normalization policy
- `configs/model.yaml`: model name and language-token mapping
- `configs/train.yaml`: hyperparameters and checkpoint strategy
- `configs/paths.yaml`: cache/output paths

## Important first-run notes

- IndicVoices is gated: accept terms in Hugging Face UI before running scripts.
- Maithili currently maps to Whisper language token `hi` in this starter setup.
- Start with small pilot steps by lowering `max_steps` in `configs/train.yaml`, then scale.

## Suggested next upgrades

- Add weighted language sampler for stronger Maithili emphasis
- Add checkpoint resume and experiment tagging
- Add decode parameter sweep per language
- Add optional W&B/TensorBoard tracking
