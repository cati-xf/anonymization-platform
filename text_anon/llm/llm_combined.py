"""
Track 3c: unlike llm_replacement.py (Track 3, always substitutes) and
llm_generalization.py (Track 3b, always generalizes), this track mirrors the
baseline pipeline's own per-category dispatch -- AnonymizationConfig.per/
loc/org/misc picks `rand` (substitution) or `gen` (generalization) per label,
same as dispatch.py's make_ne_replacement does for the rule-based pipeline
(default: PER=rand, LOC=gen, ORG=gen, MISC=tag). Entities land in exactly
one bucket by label (the config assigns one method per label, not per
entity), so:
  - the `rand`-labeled bucket goes through llm_replacement.py's substitution
    call (reused, not duplicated),
  - the `gen`-labeled bucket goes through llm_generalization.py's
    generalization call (reused, not duplicated),
  - anything left over (`sup`/`tag`/`none`) goes through
    text_anon.pipeline._assign_ne_replacements_detect() UNCHANGED --
    identical to what the baseline pipeline itself would produce for those
    entities, since the LLM never touches them.

The point of this track is a single, blended fluency report -- one
mean_delta and one mean_delta_by_method (with both "rand" and "gen" keys)
directly comparable to baseline's own results_fluency.json, rather than the
two separate single-method reports Tracks 3/3b produce.
"""
from __future__ import annotations

from ..anon_types import AnonymizationConfig, resolve_config
from ..coreference import cluster_all_labels
from ..replacement import number_duplicate_gen_replacements
from ..merge_apply import merge_and_resolve_overlaps
from ..pipeline import (
    _assign_ne_replacements_detect,
    _collect_regex_entities_detect,
    _collect_spacy_and_lexicon_entities_detect,
)

from .llm_generalization import _llm_generalizations
from .llm_replacement import _llm_replacements, _preceding_word
from .ollama_client import DEFAULT_HOST

_OPEN_LABELS = ("PER", "LOC", "ORG", "MISC")


def _spliced_replacement(text: str, start: int, end: int, res: dict) -> tuple[int, str]:
    """Same preceding-word splice llm_replacement.py/llm_generalization.py
    each do inline -- factored out here since this track needs it twice
    (once per LLM-driven bucket)."""
    core_repl = res["replacement"]
    prec = _preceding_word(text, start)
    if res["fixed_preceding_word"] and prec:
        word, word_start = prec
        return word_start, res["fixed_preceding_word"] + text[word_start + len(word):start] + core_repl
    return start, core_repl


def detect_entities_llm_combined(
    text: str,
    config: AnonymizationConfig | None = None,
    *,
    model: str,
    ollama_host: str = DEFAULT_HOST,
    local_only: bool = False,
) -> list[dict]:
    """Combined-swap variant of text_anon.pipeline.detect_entities():
    every category is still routed through whichever method the config
    assigns it, only `rand`/`gen` categories get the LLM instead of
    dispatch.py's rule-based generation."""
    cfg = resolve_config(config)

    gen_order: dict[str, list[tuple[str, tuple]]] = {}
    used_rand: dict[str, set[str]] = {}
    r_span_list, r_entity_map = _collect_regex_entities_detect(text, cfg, gen_order)
    regex_spans = [(s, e) for s, e, _, _ in r_span_list]
    number_duplicate_gen_replacements(gen_order, r_entity_map, {})

    collected_spacy, label_ctexts = _collect_spacy_and_lexicon_entities_detect(text, cfg, regex_spans)
    cluster_map = cluster_all_labels(label_ctexts)

    label_method = {label: cfg.for_spacy_label(label) for label in _OPEN_LABELS}
    rand_entities = [e for e in collected_spacy if label_method[e[1]] == "rand"]
    gen_entities = [e for e in collected_spacy if label_method[e[1]] == "gen"]
    other_entities = [e for e in collected_spacy if label_method[e[1]] not in ("rand", "gen")]

    rand_results = _llm_replacements(text, rand_entities, cluster_map, model=model, ollama_host=ollama_host)
    gen_results = _llm_generalizations(text, gen_entities, cluster_map, model=model, ollama_host=ollama_host)

    # sup/tag/none entities: identical to baseline, LLM never sees them.
    s_entity_map, _s_local_map, s_span_list = _assign_ne_replacements_detect(
        other_entities, cluster_map, label_ctexts, cfg, text, gen_order, local_only, used_rand,
    )

    # plan tuples: (eff_start, end, eff_replacement, label, source, method, core_start, core_replacement)
    plan: list[tuple] = []
    for s, e, k, nl in r_span_list:
        repl, _gender = r_entity_map[k]
        plan.append((s, e, repl, nl, "regex", cfg.regex, s, repl))

    for (orig_text, label, start, end, _origin), res in zip(rand_entities, rand_results):
        eff_start, eff_repl = _spliced_replacement(text, start, end, res)
        plan.append((eff_start, end, eff_repl, label, "llm", "rand", start, res["replacement"]))

    for (orig_text, label, start, end, _origin), res in zip(gen_entities, gen_results):
        eff_start, eff_repl = _spliced_replacement(text, start, end, res)
        plan.append((eff_start, end, eff_repl, label, "llm", "gen", start, res["replacement"]))

    for start, end, key, label, origin in s_span_list:
        repl, _gender = s_entity_map[key]
        plan.append((start, end, repl, label, origin, label_method[label], start, repl))

    kept = merge_and_resolve_overlaps(plan)

    entities: list[dict] = []
    for i, (eff_start, end, eff_repl, label, source, method, core_start, core_repl) in enumerate(kept):
        entities.append({
            "id": f"entity_{i}",
            "start": eff_start,
            "end": end,
            "original": text[eff_start:end],
            "replacement": eff_repl,
            "label": label,
            "source": source,
            "method": method,
            "local_match": True,
            "core_start": core_start,
            "core_replacement": core_repl,
        })
    return entities


def make_detector(model: str, ollama_host: str = DEFAULT_HOST, local_only: bool = False):
    def _detect(text: str, config: AnonymizationConfig | None = None) -> list[dict]:
        return detect_entities_llm_combined(text, config, model=model, ollama_host=ollama_host, local_only=local_only)
    return _detect
