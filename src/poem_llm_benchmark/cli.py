from __future__ import annotations

import argparse
import gc
import pandas as pd
from tqdm import tqdm

from .data import load_dataset, normalize_dataset, filter_split, prepare_mlx_lora_jsonl
from .eval import summarize_predictions, make_human_eval_sheet
from .mlx_runner import MLXModelRunner
from .prompts import build_prompt
from .utils import read_yaml, ensure_parent


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = read_yaml(args.config)
    model_cfg = cfg["models"]
    selected = args.models or list(model_cfg.keys())

    df = normalize_dataset(load_dataset(args.data), args.source_col, args.target_col)
    df = filter_split(df, args.split_col, args.test_split_name, args.max_examples)
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = range(len(df))

    all_rows = []
    for model_name in selected:
        if model_name not in model_cfg:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(model_cfg)}")
        spec = model_cfg[model_name]
        print(f"\nLoading {model_name}: {spec['repo']}")
        runner = MLXModelRunner(spec["repo"])
        for _, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
            prompt = build_prompt(str(row[args.source_col]), title=row.get("title"), author=row.get("author"))
            result = runner.generate(
                prompt,
                max_tokens=int(spec.get("max_tokens", args.max_tokens)),
                temperature=float(spec.get("temperature", args.temperature)),
                top_p=float(spec.get("top_p", args.top_p)),
            )
            out = row.to_dict()
            out.update({
                "model": model_name,
                "model_repo": spec["repo"],
                "prompt": prompt,
                "prediction": result.text,
                "latency_sec": result.latency_sec,
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "tokens_per_sec": result.tokens_per_sec,
            })
            all_rows.append(out)
        del runner
        gc.collect()

    pred = pd.DataFrame(all_rows)
    ensure_parent(args.out)
    pred.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote predictions: {args.out}")

    if args.summary:
        summary = summarize_predictions(pred, target_col=args.target_col)
        ensure_parent(args.summary)
        summary.to_csv(args.summary, index=False, encoding="utf-8-sig")
        print(f"Wrote summary: {args.summary}")
        print(summary)


def cmd_human_sheet(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.predictions)
    ensure_parent(args.out)
    make_human_eval_sheet(df, args.out)
    print(f"Wrote human evaluation sheet: {args.out}")


def cmd_prepare_lora_data(args: argparse.Namespace) -> None:
    df = normalize_dataset(load_dataset(args.data), args.source_col, args.target_col)
    prepare_mlx_lora_jsonl(df, args.out_dir, args.source_col, args.target_col, args.split_col)
    print(f"Wrote MLX LoRA data to: {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poem-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="Run MLX benchmark for configured models.")
    b.add_argument("--config", default="configs/models.yaml")
    b.add_argument("--data", required=True)
    b.add_argument("--models", nargs="*", default=None)
    b.add_argument("--source-col", default="source")
    b.add_argument("--target-col", default="target")
    b.add_argument("--split-col", default="split")
    b.add_argument("--test-split-name", default="test")
    b.add_argument("--max-examples", type=int, default=None)
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--temperature", type=float, default=0.2)
    b.add_argument("--top-p", type=float, default=0.9)
    b.add_argument("--out", default="results/benchmark_outputs.csv")
    b.add_argument("--summary", default="results/benchmark_summary.csv")
    b.set_defaults(func=cmd_benchmark)

    h = sub.add_parser("human-sheet", help="Create human evaluation CSV from benchmark outputs.")
    h.add_argument("--predictions", required=True)
    h.add_argument("--out", default="results/human_eval_sheet.csv")
    h.set_defaults(func=cmd_human_sheet)

    p = sub.add_parser("prepare-lora-data", help="Convert dataset to MLX LoRA JSONL files.")
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", default="data/mlx_lora")
    p.add_argument("--source-col", default="source")
    p.add_argument("--target-col", default="target")
    p.add_argument("--split-col", default="split")
    p.set_defaults(func=cmd_prepare_lora_data)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
