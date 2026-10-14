"""
Track 3: swap open-category replacement generation (dispatch.make_ne_replacement/
make_unique_rand_replacement) and grammar/article-fixing (article.fix_article)
for a single LLM call per document. Regex structured-PII detection and
replacement, plus both detection paths and coreference clustering, are reused
unmodified -- spans/clusters are therefore identical to baseline, only the
*substitution text* and its local grammar fix come from the LLM. See
LLM_MODULES_PLAN.md's Track 3 design.

The LLM is given, for every open-category mention occurrence, its original
text, category, cluster id, and (if present) the single word immediately
preceding it -- the same scope text_anon.article.fix_article ever
touches -- and asked to return a chosen replacement plus, only if needed, a
corrected version of that one preceding word. This keeps the model's edit
surface exactly as narrow as the baseline mechanism it's being compared
against, and lets the result be spliced into the same start/core_start
contract detect_entities() uses, with no text-diffing required (contrast
Track 4, which has no baseline spans to anchor to and must diff).

Regex spans keep their normal baseline replacement untouched here, with no
grammar widening applied to them by this track -- a minor fidelity gap for
the rare regex category whose value affects gender agreement (e.g. DATETIME/
QUANTITY/IDADE); out of scope for a track that specifically targets
PER/LOC/ORG/MISC substitution+grammar.
"""
from __future__ import annotations

import json
import re

from ..anon_types import AnonymizationConfig, resolve_config
from ..coreference import canonical_text, cluster_all_labels
from ..replacement import number_duplicate_gen_replacements
from ..merge_apply import merge_and_resolve_overlaps
from ..pipeline import _collect_regex_entities_detect, _collect_spacy_and_lexicon_entities_detect

from .ollama_client import chat_json, DEFAULT_HOST
from .pipeline_reuse import load_prompt

_REPLACEMENT_SCHEMA = {
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

_SYSTEM_PROMPT = load_prompt("replacement_pt.md")

_PRECEDING_WORD_RE = re.compile(r"(\S+)\s*$")


def _preceding_word(text: str, start: int) -> tuple[str, int] | None:
    """The single word immediately before `start` -- the same span
    text_anon.article.fix_article would ever rewrite. None at the start
    of the document."""
    m = _PRECEDING_WORD_RE.search(text[:start])
    return (m.group(1), m.start(1)) if m else None


def _build_occurrence_requests(
    text: str,
    collected: list[tuple[str, str, int, int, str]],
    cluster_map: dict[tuple[str, str], str],
) -> list[dict]:
    requests_ = []
    for i, (orig_text, label, start, end, _origin) in enumerate(collected):
        ctext = canonical_text(orig_text, label)
        cluster_id = f"{label}:{cluster_map.get((ctext, label), ctext)}"
        prec = _preceding_word(text, start)
        requests_.append({
            "occurrence_id": f"occ_{i}",
            "cluster_id": cluster_id,
            "label": label,
            "original_text": orig_text,
            "preceding_word": prec[0] if prec else None,
        })
    return requests_


def _llm_replacements(
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
        model=model, system=_SYSTEM_PROMPT, user=user_payload, schema=_REPLACEMENT_SCHEMA, host=ollama_host,
    )
    by_id = {
        o["id"]: o for o in response.get("occurrences", [])
        if isinstance(o, dict) and "id" in o and "replacement" in o
    }

    # Enforce Consistent Replacement (pipeline.py Sec 4.5.2's own guarantee):
    # every occurrence of the same cluster gets the FIRST replacement any of
    # its occurrences received, regardless of what the model returned for the
    # others, so a cluster can never fragment into two different replacement
    # strings within one document.
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

    # Disambiguate distinct entities that independently landed on the exact
    # same substitution text -- baseline's rand method guards against this
    # itself (dispatch.make_unique_rand_replacement retries so two different
    # real entities never share a substitute name); nothing here stops the
    # model from doing it by chance, so number collisions the same way
    # llm_generalization.py's _llm_generalizations() already does for its
    # (much more common) case of identical generalized text.
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


def detect_entities_llm_replacement(
    text: str,
    config: AnonymizationConfig | None = None,
    *,
    model: str,
    ollama_host: str = DEFAULT_HOST,
) -> list[dict]:
    """Replacement-swap variant of text_anon.pipeline.detect_entities()."""
    cfg = resolve_config(config)

    gen_order: dict[str, list[tuple[str, tuple]]] = {}
    r_span_list, r_entity_map = _collect_regex_entities_detect(text, cfg, gen_order)
    regex_spans = [(s, e) for s, e, _, _ in r_span_list]
    number_duplicate_gen_replacements(gen_order, r_entity_map, {})

    collected_spacy, label_ctexts = _collect_spacy_and_lexicon_entities_detect(text, cfg, regex_spans)
    cluster_map = cluster_all_labels(label_ctexts)

    llm_results = _llm_replacements(text, collected_spacy, cluster_map, model=model, ollama_host=ollama_host)

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
        # method="rand" (not a new "llm" value) so this passes fluency.py's/
        # human_eval.py's existing `method in ("rand", "gen")` filter without
        # touching those files -- accurate too, since this track asks the
        # model for a plausible concrete alternative (dispatch.py's `rand`
        # counterpart), not a generic category term (llm_generalization.py
        # is that counterpart, tagged "gen"). Keeping the two distinct
        # matters now that both tracks exist: fluency.py's/human_eval.py's
        # mean_delta_by_method bucketing depends on this string to compare
        # each track against its matching baseline method.
        plan.append((eff_start, end, eff_repl, label, "llm", "rand", start, core_repl))

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
        return detect_entities_llm_replacement(text, config, model=model, ollama_host=ollama_host)
    return _detect
