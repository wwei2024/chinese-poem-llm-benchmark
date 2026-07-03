#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/finetune_qwen_lora.yaml}"

python - <<'PYINNER' "$CONFIG_PATH"
import sys, yaml, subprocess
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
cmd = [
    sys.executable, "-m", "mlx_lm.lora",
    "--model", str(cfg["model"]),
    "--train",
    "--data", str(Path(cfg["train"]).parent),
    "--adapter-path", str(cfg["adapter_path"]),
    "--batch-size", str(cfg.get("batch_size", 1)),
    "--iters", str(cfg.get("iters", 300)),
    "--learning-rate", str(cfg.get("learning_rate", 1e-5)),
    "--steps-per-report", str(cfg.get("steps_per_report", 10)),
    "--steps-per-eval", str(cfg.get("steps_per_eval", 50)),
    "--save-every", str(cfg.get("save_every", 100)),
    "--max-seq-length", str(cfg.get("max_seq_length", 1024)),
    "--lora-layers", str(cfg.get("lora_layers", 16)),
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
PYINNER
