from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def normalize_dataset(df: pd.DataFrame, source_col: str = "source", target_col: str = "target") -> pd.DataFrame:
    missing = [c for c in [source_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    out = df.copy()
    out[source_col] = out[source_col].astype(str).str.strip()
    out[target_col] = out[target_col].astype(str).str.strip()
    out = out[(out[source_col] != "") & (out[target_col] != "")].reset_index(drop=True)
    return out


def filter_split(df: pd.DataFrame, split_col: str | None = "split", test_split_name: str = "test", max_examples: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if split_col and split_col in out.columns:
        selected = out[out[split_col].astype(str).str.lower() == test_split_name.lower()]
        if len(selected) > 0:
            out = selected
    if max_examples is not None:
        out = out.head(max_examples)
    return out.reset_index(drop=True)


def prepare_mlx_lora_jsonl(df: pd.DataFrame, out_dir: str | Path, source_col: str = "source", target_col: str = "target", split_col: str = "split") -> None:
    from .prompts import build_prompt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split_col in df.columns:
        split = df[split_col].astype(str).str.lower()
        train = df[split.eq("train")]
        valid = df[split.isin(["valid", "validation", "dev"])]
    else:
        train = df
        valid = df.head(min(10, len(df)))
    if len(train) == 0:
        train = df
    if len(valid) == 0:
        valid = train.head(min(10, len(train)))

    def write_jsonl(part: pd.DataFrame, filename: str) -> None:
        with (out_dir / filename).open("w", encoding="utf-8") as f:
            for _, row in part.iterrows():
                prompt = build_prompt(str(row[source_col]), title=row.get("title"), author=row.get("author"))
                record = {"text": prompt + str(row[target_col])}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl(train, "train.jsonl")
    write_jsonl(valid, "valid.jsonl")
