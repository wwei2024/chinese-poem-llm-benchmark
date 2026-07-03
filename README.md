# Chinese Poem LLM Benchmark

Benchmark and fine-tune instruction models for **classical Chinese poem understanding** on Apple Silicon, especially an **M4 Mac mini with 32 GB unified memory**.

The package compares two model families:

- **Qwen**: strong Chinese-language baseline
- **Gemma 4**: newer Google model family, tested through MLX-compatible 4-bit checkpoints

The project supports:

- MLX-based local inference benchmarking
- side-by-side model comparison
- ROUGE-L, BLEU, optional BERTScore evaluation
- latency and tokens/sec measurement
- human evaluation sheet generation
- LoRA fine-tuning scaffolding through `mlx_lm.lora`
- reproducible configs for dataset/model paths

---

## 1. Recommended environment

Use a clean environment. Do **not** reuse a PaddleX/PaddleOCR environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

This project pins `numpy<2.4` to avoid common conflicts with packages such as PaddleX.

---

## 2. Default models

```yaml
qwen_7b_4bit: mlx-community/Qwen2.5-7B-Instruct-4bit
gemma4_12b_4bit: mlx-community/gemma-4-12B-it-4bit
```

On a 32 GB M4 Mac mini, benchmark one model at a time. Treat Gemma 4 12B 4-bit as an inference benchmark first; for local fine-tuning, start with Qwen 7B or a smaller Gemma checkpoint.

---

## 3. Dataset format

Your dataset should contain at least:

| column | meaning |
|---|---|
| `source` | classical Chinese poem line or poem text |
| `target` | modern Chinese explanation / restatement |

Optional: `title`, `author`, `split`.

Supported files: `.csv`, `.xlsx`, `.json`, `.jsonl`.

A tiny sample is included at `data/sample/poems_sample.csv`.

---

## 4. Run a benchmark

```bash
poem-bench benchmark   --config configs/models.yaml   --data data/sample/poems_sample.csv   --out results/benchmark_outputs.csv   --summary results/benchmark_summary.csv
```

Run only one model:

```bash
poem-bench benchmark   --config configs/models.yaml   --data data/sample/poems_sample.csv   --models qwen_7b_4bit   --out results/qwen_outputs.csv
```

---

## 5. Generate a human evaluation sheet

```bash
poem-bench human-sheet   --predictions results/benchmark_outputs.csv   --out results/human_eval_sheet.csv
```

Rubric suggestion: score accuracy, fluency, imagery preservation, and hallucination from 1 to 5.

---

## 6. Fine-tune with MLX LoRA

Prepare train/validation JSONL files:

```bash
poem-bench prepare-lora-data   --data your_dataset.csv   --out-dir data/mlx_lora
```

Then start LoRA fine-tuning:

```bash
bash scripts/train_lora_mlx.sh configs/finetune_qwen_lora.yaml
```

This wrapper calls `mlx_lm.lora`. Review the config before running. On 32 GB Apple Silicon, begin with Qwen 7B 4-bit or a smaller Gemma model. Do not start with full fine-tuning of a 12B model.

---

## 7. Repository layout

```text
configs/                 model and training configs
data/sample/             tiny sample dataset
notebooks/               copied benchmark notebook and archived previous attempt
scripts/                 shell wrappers
src/poem_llm_benchmark/  reusable Python package
results/                 generated outputs
adapters/                generated LoRA adapters
```

---

## 8. Practical recommendation

1. Benchmark **Qwen 7B 4-bit** vs **Gemma 4 12B 4-bit** without fine-tuning.
2. Inspect side-by-side outputs on hard poems.
3. Fine-tune **Qwen 7B LoRA** first.
4. Only then try smaller/adapter-based Gemma fine-tuning.

Chinese poem understanding is not only translation. Evaluate factual meaning, imagery preservation, tone, and hallucination separately.
