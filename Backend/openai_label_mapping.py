import json
import os
from pathlib import Path
from typing import Dict, List

TRASH = "trash"
RECYCLING = "recycling"
VALID_BINARY = {TRASH, RECYCLING}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def keyword_fallback(label: str) -> str:
    s = label.lower()
    recycling_tokens = {
        "bottle",
        "can",
        "glass",
        "metal",
        "paper",
        "cardboard",
        "plastic",
        "carton",
    }
    for token in recycling_tokens:
        if token in s:
            return RECYCLING
    return TRASH


def llm_map_label_openai(label: str, model_name: str) -> str:
    """
    Maps one raw TACO class label to a binary class using OpenAI.
    Requires OPENAI_API_KEY in environment.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = (
        "You map waste item labels to one of two classes: 'trash' or 'recycling'. "
        "Return only one word: trash or recycling."
    )
    user_prompt = (
        f"Label: {label}\n"
        "Classify it as either trash or recycling for a practical curbside sorting app."
    )
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    text = (resp.output_text or "").strip().lower()
    if text in VALID_BINARY:
        return text
    if TRASH in text:
        return TRASH
    if RECYCLING in text:
        return RECYCLING
    raise ValueError(f"Unexpected LLM label mapping output for '{label}': {text!r}")


def build_binary_label_map(
    category_names: List[str],
    cache_path: Path,
    openai_model: str,
) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if cache_path.exists():
        cache = _load_json(cache_path)

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    for raw_label in category_names:
        if raw_label in cache and cache[raw_label] in VALID_BINARY:
            continue
        if has_openai:
            try:
                mapped = llm_map_label_openai(raw_label, openai_model)
            except Exception:
                mapped = keyword_fallback(raw_label)
        else:
            mapped = keyword_fallback(raw_label)
        cache[raw_label] = mapped

    _save_json(cache_path, cache)
    return {k: v for k, v in cache.items() if k in category_names}
