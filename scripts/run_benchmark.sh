#!/usr/bin/env bash
set -euo pipefail

poem-bench benchmark   --config configs/models.yaml   --data "${1:-data/sample/poems_sample.csv}"   --out results/benchmark_outputs.csv   --summary results/benchmark_summary.csv

poem-bench human-sheet   --predictions results/benchmark_outputs.csv   --out results/human_eval_sheet.csv
