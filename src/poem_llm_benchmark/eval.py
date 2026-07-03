from __future__ import annotations

import pandas as pd


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return float(sum(scores) / max(len(scores), 1))


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    import sacrebleu
    return float(sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score)


def compute_bertscore(predictions: list[str], references: list[str]) -> float | None:
    try:
        from bert_score import score
    except Exception:
        return None
    _, _, f1 = score(predictions, references, lang="zh", verbose=False)
    return float(f1.mean().item())


def summarize_predictions(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    rows = []
    for model_name, group in df.groupby("model"):
        preds = group["prediction"].fillna("").astype(str).tolist()
        refs = group[target_col].fillna("").astype(str).tolist()
        row = {
            "model": model_name,
            "n": len(group),
            "rouge_l": compute_rouge_l(preds, refs),
            "bleu": compute_bleu(preds, refs),
            "avg_latency_sec": group["latency_sec"].mean(),
            "avg_tokens_per_sec": group["tokens_per_sec"].mean(),
        }
        bert = compute_bertscore(preds, refs)
        if bert is not None:
            row["bertscore_f1"] = bert
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rouge_l", ascending=False)


def make_human_eval_sheet(predictions: pd.DataFrame, out_path: str) -> None:
    cols = [c for c in ["id", "title", "author", "source", "target", "model", "prediction"] if c in predictions]
    sheet = predictions[cols].copy()
    sheet["accuracy_1_5"] = ""
    sheet["fluency_1_5"] = ""
    sheet["imagery_1_5"] = ""
    sheet["hallucination_1_5"] = ""
    sheet["notes"] = ""
    sheet.to_csv(out_path, index=False, encoding="utf-8-sig")
