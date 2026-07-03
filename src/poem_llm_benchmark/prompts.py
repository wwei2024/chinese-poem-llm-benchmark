SYSTEM_PROMPT = """你是一位严谨的中国古典诗歌解释助手。请把古典诗句解释成自然、准确的现代汉语。不要添加原诗没有的信息。"""


def build_prompt(source: str, title: str | None = None, author: str | None = None) -> str:
    meta = []
    if title:
        meta.append(f"题目：{title}")
    if author:
        meta.append(f"作者：{author}")
    meta_text = "\n".join(meta)
    if meta_text:
        meta_text += "\n"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{meta_text}原文：{source}\n\n"
        "任务：请用现代汉语解释这句诗的意思，要求准确、简洁、保留意象。\n"
        "现代汉语解释："
    )
