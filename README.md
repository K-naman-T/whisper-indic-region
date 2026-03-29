# Whisper Indic Fine-Tuning

Fine-tuning OpenAI's Whisper model for Indian regional languages using the `ai4bharat/IndicVoices` dataset.

## Training Plan: Phase 1 (Multi-Lingual Pilot)

### 1. Infrastructure Setup
*   **Target Node**: `vyom-node-3` (RTX 4090).
*   **Environment**: Python 3.10+ virtual environment.
*   **Core Dependencies**: `torch`, `transformers`, `peft` (LoRA), `bitsandbytes`, `datasets`, `accelerate`.

### 2. Data Strategy
*   **Dataset**: `ai4bharat/IndicVoices` (22 Languages).
*   **Target Languages**:
    *   **Hindi** (Devanagari)
    *   **Bangla** (Bengali)
    *   **Maithili** (Devanagari)
    *   **Urdu** (Nastaliq/Arabic)
*   **Preprocessing**:
    *   Resampling: 16kHz mono.
    *   Normalization: Unicode NFC, whitespace stripping, and script-aware character cleaning.
    *   Sampling: Weighted sampling to ensure Maithili and Urdu represent equal gradient contribution.

### 3. Model Configuration
*   **Base Model**: `openai/whisper-medium`.
*   **Fine-tuning Method**: **LoRA (PEFT)**.
    *   `r=16`, `lora_alpha=32`.
    *   Targets: `q_proj`, `v_proj`.
*   **Optimization**: 8-bit quantization (`bitsandbytes`) to maximize batch size on RTX 4090.
*   **Hyperparameters**:
    *   LR: `1e-4`.
    *   Batch Size: `16` (Effective `32` with grad accumulation).
    *   Scheduler: Linear warmup + Cosine decay.

### 4. Execution Workflow
1.  **Environment Check**: `python scripts/doctor.py`
2.  **Data Preparation**: `python scripts/prepare_data.py --languages hi,bn,mai,ur`
3.  **Training**: `python scripts/train.py --config configs/train.yaml`
4.  **Evaluation**: `python scripts/evaluate.py` targeting WER (Word Error Rate) and CER (Character Error Rate).

## Repository Structure
*   `configs/`: YAML configurations for data, model, and training.
*   `scripts/`: Core execution scripts (train, prepare, evaluate).
*   `src/`: Shared source code (data loaders, metrics, utils).
*   `outputs/`: (Ignored) Model checkpoints, logs, and reports.

## Streaming Plan (Low Disk Mode)

We will use Hugging Face dataset streaming to avoid full local downloads on E:. The pipeline will load IndicVoices with `streaming=True`, normalize and filter samples on the fly, and keep training step-based.

### Streaming Workflow Updates
1.  **Config Updates**
    - Add streaming fields in `configs/data.yaml` (streaming flag, audio/text column names, max eval samples, and split strategy).
    - Keep `configs/paths.yaml` cache_dir for HF streaming cache.
2.  **Streaming Loader**
    - New `src/data/streaming.py` to load HF splits per language with `streaming=True`.
    - Apply normalization using `src/text/normalize.normalize_text` and filters before training.
    - Interleave languages using `language_weights` for balanced sampling.
3.  **Training**
    - `scripts/train.py` will consume streaming splits directly (no parquet manifests).
    - Evaluation will materialize a small in-memory set via `max_eval_samples`.
4.  **Evaluation & Audit**
    - `scripts/evaluate.py` and `scripts/audit_dataset.py` will use streaming with sample limits.
5.  **Doctor Check**
    - `scripts/doctor.py` will validate HF access using streaming (`take(1)`) and relax disk-space checks.

### Notes
- Requires `HF_TOKEN` to access IndicVoices.
- Streaming avoids large local downloads but still uses cache (controlled by `configs/paths.yaml`).

## OpenCode Prompt
> "Clone `https://github.com/K-naman-T/whisper-indic-region`. Set up a Python venv and install requirements. Use the ML Engineer skill to configure a fine-tuning run for `whisper-medium` on Hindi, Bangla, Maithili, and Urdu. Implement LoRA (PEFT) for parameter-efficient training. Run `scripts/prepare_data.py` then initiate `scripts/train.py` on the RTX 4090."
