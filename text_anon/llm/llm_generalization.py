"""
Track 3b: swap open-category replacement generation (dispatch.generalize_ne)
and grammar/article-fixing (article.fix_article) for a single LLM call per
document -- the generalization counterpart to llm_replacement.py's
substitution track. Regex structured-PII detection, both detection paths,
and coreference clustering are reused unmodified -- spans/clusters are
therefore identical to baseline, only the *generalization text* and its
local grammar fix come from the LLM. Mirrors llm_replacement.py's structure
and its Track 3 design in LLM_MODULES_PLAN.md closely; see that file's
docstring for the shared occurrence-request shape and the reasoning behind
scoping the model's edit surface to one preceding word.

Unlike substitution, where each occurrence can plausibly get a different
concrete replacement, generalization collapses distinct entities of the same
category towards the same generic term (e.g. two different people both
becoming "pessoa anónima") -- exactly what dispatch.number_duplicate_gen_replacements
guards against for the rule-based `gen` method. _llm_generalizations()
replicates that guarantee here: after the model's per-occurrence choices are
resolved to one replacement per cluster, clusters that landed on the
identical generalized text get numbered, deterministically in code rather
than left to the model to self-manage.
"""
from __future__ import annotations

import json

from ..anon_types import AnonymizationConfig, resolve_config
from ..coreference import canonical_text, cluster_all_labels
from ..replacement import number_duplicate_gen_replacements
from ..merge_apply import merge_and_resolve_overlaps
from ..pipeline import _collect_regex_entities_detect, _collect_spacy_and_lexicon_entities_detect

from .llm_replacement import _build_occurrence_requests, _preceding_word
from .ollama_client import chat_json, DEFAULT_HOST
from .pipeline_reuse import load_prompt

_GENERALIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "occurrences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "replacement": {"type": "string"},
                    "fixed_preceding_word": {"type": ["string", "null"]},
                },
                "required": ["id", "replacement", "fixed_preceding_word"],
            },
        },
    },
    "required": ["occurrences"],
}

_SYSTEM_PROMPT = load_prompt("generalization_pt.md")


def _llm_generalizations(
    text: str,
    collected: list[tuple[str, str, int, int, str]],
    cluster_map: dict[tuple[str, str], str],
    *,
    model: str,
    ollama_host: str,
) -> list[dict]:
    occ_requests = _build_occurrence_requests(text, collected, cluster_map)
    if not occ_requests:
        return []
    user_payload = json.dumps({"document": text, "occurrences": occ_requests}, ensure_ascii=False)
    response = chat_json(
        model=model, system=_SYSTEM_PROMPT, user=user_payload, schema=_GENERALIZATION_SCHEMA, host=ollama_host,
    )
    by_id = {
        o["id"]: o for o in response.get("occurrences", [])
        if isinstance(o, dict) and "id" in o and "replacement" in o
    }

    # Consistent Replacement (pipeline.py Sec 4.5.2's own guarantee): every
    # occurrence of the same cluster gets the FIRST replacement any of its
    # occurrences received.
    cluster_choice: dict[str, str] = {}
    cluster_order: list[str] = []
    label_counters: dict[str, int] = {}
    for req in occ_requests:
        answer = by_id.get(req["occurrence_id"], {})
        cluster_id = req["cluster_id"]
        if cluster_id not in cluster_choice:
            repl = answer.get("replacement")
            if not repl:
                # Model omitted this occurrence (or returned an empty
                # string) -- never fall back to the original identifying
                # text here: for an anonymization system that would
                # silently leak PII into the "anonymized" output. Fall back
                # to the same safe bracket-tag placeholder baseline's own
                # "tag" method uses (dispatch.py) instead.
                label = req["label"]
                label_counters[label] = label_counters.get(label, 0) + 1
                repl = f"[{label}_{label_counters[label]}]"
            cluster_choice[cluster_id] = repl
            cluster_order.append(cluster_id)

    # Disambiguate distinct entities the model generalized to the identical
    # term -- see module docstring. Scoped by label as well as text: two
    # different categories landing on coincidentally identical generic text
    # isn't the collision this guards against.
    by_text: dict[tuple[str, str], list[str]] = {}
    for cid in cluster_order:
        label = cid.split(":", 1)[0]
        by_text.setdefault((label, cluster_choice[cid]), []).append(cid)
    for (_label, base_text), cids in by_text.items():
        if len(cids) > 1:
            for idx, cid in enumerate(cids, 1):
                cluster_choice[cid] = f"{base_text} {idx}"

    results = []
    for req in occ_requests:
        answer = by_id.get(req["occurrence_id"], {})
        results.append({
            **req,
            "replacement": cluster_choice[req["cluster_id"]],
            "fixed_preceding_word": answer.get("fixed_preceding_word"),
        })
    return results


def detect_entities_llm_generalization(
    text: str,
    config: AnonymizationConfig | None = None,
    *,
    model: str,
    ollama_host: str = DEFAULT_HOST,
) -> list[dict]:
    """Generalization-swap variant of text_anon.pipeline.detect_entities()."""
    cfg = resolve_config(config)

    gen_order: dict[str, list[tuple[str, tuple]]] = {}
    r_span_list, r_entity_map = _collect_regex_entities_detect(text, cfg, gen_order)
    regex_spans = [(s, e) for s, e, _, _ in r_span_list]
    number_duplicate_gen_replacements(gen_order, r_entity_map, {})

    collected_spacy, label_ctexts = _collect_spacy_and_lexicon_entities_detect(text, cfg, regex_spans)
    cluster_map = cluster_all_labels(label_ctexts)

    llm_results = _llm_generalizations(text, collected_spacy, cluster_map, model=model, ollama_host=ollama_host)

    # plan tuples: (eff_start, end, eff_replacement, label, source, method, core_start, core_replacement)
    plan: list[tuple] = []
    for s, e, k, nl in r_span_list:
        repl, _gender = r_entity_map[k]
        plan.append((s, e, repl, nl, "regex", cfg.regex, s, repl))

    for (orig_text, label, start, end, _origin), res in zip(collected_spacy, llm_results):
        core_repl = res["replacement"]
        prec = _preceding_word(text, start)
        if res["fixed_preceding_word"] and prec:
            word, word_start = prec
            eff_start = word_start
            eff_repl = res["fixed_preceding_word"] + text[word_start + len(word):start] + core_repl
        else:
            eff_start = start
            eff_repl = core_repl
        # method="gen" is accurate here (this track *is* the LLM-driven
        # generalization counterpart to dispatch.generalize_ne), and lets
        # this pass fluency.py's/human_eval.py's existing method in
        # ("rand", "gen") filter unchanged.
        plan.append((eff_start, end, eff_repl, label, "llm", "gen", start, core_repl))

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


def make_detector(model: str, ollama_host: str = DEFAULT_HOST):
    def _detect(text: str, config: AnonymizationConfig | None = None) -> list[dict]:
        return detect_entities_llm_generalization(text, config, model=model, ollama_host=ollama_host)
    return _detect
