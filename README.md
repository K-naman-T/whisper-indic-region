# Whisper Indic Fine-Tuning

Fine-tuning OpenAI's Whisper model for Indian regional languages using the `ai4bharat/IndicVoices` dataset.

## Training Plan: Phase 1 (Pilot)

### 1. Infrastructure Setup
*   **Target Node**: `vyom-node-3` (RTX 4090).
*   **Environment**: Python 3.10+ virtual environment.
*   **Core Dependencies**: `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `accelerate`.

### 2. Data Strategy
*   **Dataset**: `ai4bharat/IndicVoices` (Multilingual Speech Dataset).
*   **Target Languages**: Hindi, Bengali, Tamil.
*   **Preprocessing**:
    *   Audio resampling to 16kHz mono.
    *   Unicode normalization for transcripts.
    *   Quality filtering (dropping noisy/short clips).

### 3. Model Configuration
*   **Base Model**: `openai/whisper-medium`.
*   **Fine-tuning Method**: **LoRA (PEFT)**.
    *   Enables parameter-efficient tuning.
    *   Reduced VRAM footprint on RTX 4090.
*   **Optimization**: 8-bit quantization (`bitsandbytes`) and mixed-precision (FP16/BF16).

### 4. Execution Workflow
1.  **Environment Check**: Run `python scripts/doctor.py` to verify GPU/CUDA availability.
2.  **Data Preparation**: Run `python scripts/prepare_data.py` for the selected languages.
3.  **Training**: Execute `python scripts/train.py` using configurations in `configs/train.yaml`.
4.  **Evaluation**: Run `python scripts/evaluate.py` to calculate WER/CER per language.

## Repository Structure
*   `configs/`: YAML configurations for data, model, and training.
*   `scripts/`: Core execution scripts (train, prepare, evaluate).
*   `src/`: Shared source code (data loaders, metrics, utils).
*   `outputs/`: (Ignored) Model checkpoints, logs, and reports.

## OpenCode Prompt
> "Clone the repository `https://github.com/K-naman-T/whisper-indic-region`. Initialize a virtual environment and install dependencies from `requirements.txt`. Configure the system to fine-tune `whisper-medium` using LoRA (PEFT) on the `ai4bharat/IndicVoices` dataset. Target languages: Hindi, Bengali, and Tamil. Run the `scripts/prepare_data.py` script to verify the pipeline, then execute `scripts/train.py` utilizing the RTX 4090 for hardware acceleration."
