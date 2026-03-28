#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "doctor_report.json"


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


def _run_check(name: str, fn: Callable[[], tuple[str, str]]) -> CheckResult:
    try:
        status, details = fn()
    except Exception as exc:
        status, details = "FAIL", f"Unhandled exception: {exc}"
    return CheckResult(name=name, status=status, details=details)


def check_python_version() -> tuple[str, str]:
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) >= (3, 10):
        return "PASS", f"Python {major}.{minor}"
    return "FAIL", f"Python {major}.{minor} found; require >= 3.10"


def check_venv() -> tuple[str, str]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        return "PASS", f"Virtual environment active: {sys.prefix}"
    return "WARN", "No virtual environment detected"


def check_system_tools() -> tuple[str, str]:
    required = ["git", "ffmpeg", "nvidia-smi"]
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        return "FAIL", f"Missing tools: {', '.join(missing)}"
    return "PASS", "All required system tools found"


def check_disk_space() -> tuple[str, str]:
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024**3)
    if free_gb >= 300:
        return "PASS", f"Free disk space: {free_gb:.1f} GB"
    if free_gb >= 150:
        return "WARN", f"Free disk space: {free_gb:.1f} GB (recommended >= 300 GB)"
    return "FAIL", f"Free disk space too low: {free_gb:.1f} GB"


def check_torch_cuda() -> tuple[str, str]:
    try:
        import torch
    except Exception as exc:
        return "FAIL", f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return "FAIL", "torch.cuda.is_available() is False"

    count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(count)]
    memory_gb = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(count)]
    gpu_summary = "; ".join(f"{name} ({mem:.1f} GB)" for name, mem in zip(gpu_names, memory_gb))
    return "PASS", f"CUDA available on {count} GPU(s): {gpu_summary}"


def check_python_packages() -> tuple[str, str]:
    required = [
        "datasets",
        "evaluate",
        "transformers",
        "accelerate",
        "torchaudio",
        "soundfile",
        "yaml",
        "rich",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        return "FAIL", f"Missing python packages: {', '.join(missing)}"
    return "PASS", "All required python packages import successfully"


def check_hf_token() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return "PASS", "HF_TOKEN is set"
    return "FAIL", "HF_TOKEN is not set"


def check_hf_dataset_access() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "WARN", "Skipped dataset access check because HF_TOKEN is unset"

    try:
        from datasets import load_dataset
    except Exception as exc:
        return "FAIL", f"datasets import failed: {exc}"

    try:
        ds = load_dataset(
            "ai4bharat/IndicVoices",
            "hindi",
            split="valid[:1]",
            token=token,
        )
        sample_count = len(ds)
        return "PASS", f"IndicVoices access OK (retrieved {sample_count} sample(s))"
    except Exception as exc:
        return "FAIL", f"IndicVoices access failed: {exc}"


def check_output_paths() -> tuple[str, str]:
    target_dirs = [
        PROJECT_ROOT / "outputs" / "logs",
        PROJECT_ROOT / "outputs" / "reports",
        PROJECT_ROOT / "outputs" / "checkpoints",
        PROJECT_ROOT / "outputs" / "manifests",
    ]
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        test_file = directory / ".write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
        except Exception as exc:
            return "FAIL", f"Cannot write to {directory}: {exc}"
    return "PASS", "Output directories are writable"


def check_nvidia_smi() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return "FAIL", "nvidia-smi is not installed"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if not output:
            return "FAIL", "nvidia-smi returned empty output"
        return "PASS", output
    except Exception as exc:
        return "FAIL", f"nvidia-smi query failed: {exc}"


def main() -> int:
    checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("python_version", check_python_version),
        ("virtual_environment", check_venv),
        ("system_tools", check_system_tools),
        ("disk_space", check_disk_space),
        ("python_packages", check_python_packages),
        ("torch_cuda", check_torch_cuda),
        ("nvidia_smi", check_nvidia_smi),
        ("hf_token", check_hf_token),
        ("hf_dataset_access", check_hf_dataset_access),
        ("output_paths", check_output_paths),
    ]

    results = [_run_check(name, fn) for name, fn in checks]
    status_order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    results.sort(key=lambda item: status_order.get(item.status, 3))

    print(f"System: {platform.platform()}")
    print(f"Project: {PROJECT_ROOT}")
    print("\nDoctor Report")
    print("=" * 72)
    for result in results:
        print(f"[{result.status:<4}] {result.name:<20} - {result.details}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "results": [asdict(r) for r in results],
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report to {REPORT_PATH}")

    has_failures = any(result.status == "FAIL" for result in results)
    if has_failures:
        print("\nDoctor completed with FAIL status. Fix failing checks and rerun.")
        return 1

    print("\nDoctor completed successfully (PASS/WARN only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
