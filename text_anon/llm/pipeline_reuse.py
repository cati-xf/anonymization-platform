"""
Shared finalization logic reused by llm_detection.py and llm_coreference.py --
both compose text_anon.pipeline's own private per-stage helpers with
exactly one stage swapped, and both need to turn the resulting intermediates
into the same list[dict] shape text_anon.pipeline.detect_entities()
returns. Factored out here rather than duplicated in both files -- see
pipeline.py's detect_entities() for the baseline this mirrors line for line.

Also holds the prompts/ loader shared by every track.
"""
from __future__ import annotations

from pathlib import Path

from ..merge_apply import merge_and_resolve_overlaps, simulate_article_fix

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def finalize_detect_entities(
    text: str,
    r_span_list: list[tuple[int, int, tuple, str]],
    r_entity_map: dict[tuple, tuple],
    regex_method: str,
    s_span_list: list[tuple[int, int, tuple, str, str]],
    s_entity_map: dict[tuple, tuple],
    s_local_map: dict[tuple, tuple],
) -> list[dict]:
    """Mirrors pipeline.detect_entities()'s own final-assembly block exactly --
    merge, simulate the article/grammar fix, and build the reported dicts."""
    plan_regex = [(s, e, *r_entity_map[k], nl, "regex") for s, e, k, nl in r_span_list]
    plan_spacy = [(s, e, *s_entity_map[k], nl, origin) for s, e, k, nl, origin in s_span_list]

    span_local: dict[tuple[int, int], bool] = {}
    span_method: dict[tuple[int, int], str] = {}
    for s, e, k, nl in r_span_list:
        span_local[(s, e)] = True
        span_method[(s, e)] = regex_method
    for s, e, k, nl, origin in s_span_list:
        had_local, method = s_local_map.get(k, (True, "tag"))
        span_local[(s, e)] = had_local
        span_method[(s, e)] = method

    kept = merge_and_resolve_overlaps(plan_regex + plan_spacy)
    art_info = simulate_article_fix(text, kept)

    entities: list[dict] = []
    for i, (start, end, repl, gender, label, source) in enumerate(kept):
        art = art_info.get(i)
        if art:
            art_start, art_repl = art
            eff_start, eff_original, eff_replacement = art_start, text[art_start:end], art_repl + repl
        else:
            eff_start, eff_original, eff_replacement = start, text[start:end], repl

        entities.append({
            "id": f"entity_{i}",
            "start": eff_start,
            "end": end,
            "original": eff_original,
            "replacement": eff_replacement,
            "label": label,
            "source": source,
            "method": span_method.get((start, end), "tag"),
            "local_match": span_local.get((start, end), True),
            "core_start": start,
            "core_replacement": repl,
        })
    return entities
