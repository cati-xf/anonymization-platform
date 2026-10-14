import re

from .anon_types import AnonymizationConfig, AnonSpan, resolve_config, whitelist_match, whitelist_contains
from .detection import (
    find_all as _find_regex_matches,
    nlp,
    mask_titles,
    build_masked_to_orig,
    find_lexicon_person_candidates,
)
from .coreference import canonical_text, cluster_all_labels
from .replacement import (
    make_regex_replacement,
    make_ne_replacement,
    make_unique_rand_replacement,
    number_duplicate_gen_replacements,
    lookup_live,
    infer_person_gender,
)
from .merge_apply import (
    merge_and_resolve_overlaps,
    apply_with_article_fix,
    simulate_article_fix,
    apply_entities,
)
from .article import _TRANSIENT_GENDER


# ── regex detection + replacement assignment ─────────────────────────────────

def _collect_regex_entities(
    text: str,
    config: AnonymizationConfig,
    gen_order: dict[str, list[tuple[str, tuple]]],
) -> tuple[list[tuple[int, int, tuple]], dict[tuple, tuple]]:
    """
    Detects regex spans, filters by config (skip labels / whitelist), dedupes by
    (label, normalized text) key, and assigns each unique key its replacement.
    Returns (r_span_list, r_entity_map); appends to gen_order in place.
    """
    regex_matches = _find_regex_matches(text)
    r_counters: dict[str, int] = {}
    r_entity_map: dict[tuple, tuple] = {}
    r_span_list: list[tuple[int, int, tuple]] = []

    for m in regex_matches:
        if config.regex == "none" or m.label in config.regex_skip_labels:
            continue
        norm_text = re.sub(r"\s+", " ", m.text.strip().lower())
        if whitelist_match(norm_text, config.whitelist) or whitelist_contains(norm_text, config.whitelist):
            continue
        norm = re.sub(r"\s+", "", m.text.lower())
        key = (m.label, norm)
        if key not in r_entity_map:
            r_counters[m.label] = r_counters.get(m.label, 0) + 1
            repl, gender = make_regex_replacement(
                m.label, m.text, r_counters[m.label], config.regex
            )
            if config.regex == "gen" and not re.fullmatch(r"\d{1,4}", repl):
                gen_order.setdefault(repl, []).append(("r", key))
            r_entity_map[key] = (repl, gender)
        r_span_list.append((m.start, m.end, key))

    return r_span_list, r_entity_map


def _collect_regex_entities_detect(
    text: str,
    cfg: AnonymizationConfig,
    gen_order: dict[str, list[tuple[str, tuple]]],
) -> tuple[list[tuple[int, int, tuple, str]], dict[tuple, tuple]]:
    """Same as _collect_regex_entities, but keeps the ner label alongside each span."""
    regex_matches = _find_regex_matches(text)
    r_counters: dict[str, int] = {}
    r_entity_map: dict[tuple, tuple] = {}
    r_span_list: list[tuple[int, int, tuple, str]] = []

    for m in regex_matches:
        if cfg.regex == "none" or m.label in cfg.regex_skip_labels:
            continue
        norm_text = re.sub(r"\s+", " ", m.text.strip().lower())
        if whitelist_match(norm_text, cfg.whitelist) or whitelist_contains(norm_text, cfg.whitelist):
            continue
        norm = re.sub(r"\s+", "", m.text.lower())
        key = (m.label, norm)
        if key not in r_entity_map:
            r_counters[m.label] = r_counters.get(m.label, 0) + 1
            repl, gender = make_regex_replacement(m.label, m.text, r_counters[m.label], cfg.regex)
            if cfg.regex == "gen" and not re.fullmatch(r"\d{1,4}", repl):
                gen_order.setdefault(repl, []).append(("r", key))
            r_entity_map[key] = (repl, gender)
        r_span_list.append((m.start, m.end, key, m.label))

    return r_span_list, r_entity_map


# ── spaCy + lexicon detection ─────────────────────────────────────────────────

def _collect_spacy_and_lexicon_entities(
    text: str,
    config: AnonymizationConfig,
    regex_spans: list[tuple[int, int]],
) -> tuple[list[tuple[str, str, int, int]], dict[str, dict[str, str]]]:
    """
    Runs spaCy NER (after title-masking) plus lexicon gap-fill for PER names
    spaCy missed. Returns (collected_spacy, label_ctexts) where collected_spacy
    is [(orig_text, label, orig_start, orig_end)] and label_ctexts is
    {label: {canonical_text: best_original_text}}.
    """
    masked_text, title_reps = mask_titles(text)
    offset_map = build_masked_to_orig(text, title_reps)

    doc = nlp(masked_text)
    collected_spacy: list[tuple[str, str, int, int]] = []
    label_ctexts: dict[str, dict[str, str]] = {}

    for ent in doc.ents:
        orig_start = offset_map[ent.start_char]
        orig_end = offset_map[min(ent.end_char, len(offset_map) - 1)]
        if any(rs < orig_end and re_ > orig_start for rs, re_ in regex_spans):
            continue
        orig_text = text[orig_start:orig_end]
        label = ent.label_
        if config.for_spacy_label(label) == "none":
            continue
        ctext = canonical_text(orig_text, label)
        norm_text = orig_text.strip().lower()
        if (
            whitelist_match(norm_text, config.whitelist) or whitelist_contains(norm_text, config.whitelist)
            or whitelist_match(ctext, config.whitelist) or whitelist_contains(ctext, config.whitelist)
        ):
            continue
        ct = label_ctexts.setdefault(label, {})
        if ctext not in ct or len(orig_text) > len(ct[ctext]):
            ct[ctext] = orig_text  # keep longest original for replacement generation
        collected_spacy.append((orig_text, label, orig_start, orig_end))

    # Lexicon gap-fill: catch PER names missed by spacy/regex via name/surname dictionaries
    if config.for_spacy_label("PER") != "none":
        covered_spans = regex_spans + [(s, e) for _, _, s, e in collected_spacy]
        for cand_start, cand_end, cand_text in find_lexicon_person_candidates(text):
            if any(cs < cand_end and ce > cand_start for cs, ce in covered_spans):
                continue
            ctext = canonical_text(cand_text, "PER")
            norm_text = cand_text.strip().lower()
            if (
                whitelist_match(norm_text, config.whitelist) or whitelist_contains(norm_text, config.whitelist)
                or whitelist_match(ctext, config.whitelist) or whitelist_contains(ctext, config.whitelist)
            ):
                continue
            ct = label_ctexts.setdefault("PER", {})
            if ctext not in ct or len(cand_text) > len(ct[ctext]):
                ct[ctext] = cand_text
            collected_spacy.append((cand_text, "PER", cand_start, cand_end))

    return collected_spacy, label_ctexts


def _collect_spacy_and_lexicon_entities_detect(
    text: str,
    cfg: AnonymizationConfig,
    regex_spans: list[tuple[int, int]],
) -> tuple[list[tuple[str, str, int, int, str]], dict[str, dict[str, str]]]:
    """Same as _collect_spacy_and_lexicon_entities, but tags each entity with its origin ("spacy"/"lexicon")."""
    masked_text, title_reps = mask_titles(text)
    offset_map = build_masked_to_orig(text, title_reps)
    doc = nlp(masked_text)

    collected_spacy: list[tuple[str, str, int, int, str]] = []
    label_ctexts: dict[str, dict[str, str]] = {}

    for ent in doc.ents:
        orig_start = offset_map[ent.start_char]
        orig_end = offset_map[min(ent.end_char, len(offset_map) - 1)]
        if any(rs < orig_end and re_ > orig_start for rs, re_ in regex_spans):
            continue
        orig_text = text[orig_start:orig_end]
        label = ent.label_
        if cfg.for_spacy_label(label) == "none":
            continue
        ctext = canonical_text(orig_text, label)
        norm_text = orig_text.strip().lower()
        if (
            whitelist_match(norm_text, cfg.whitelist) or whitelist_contains(norm_text, cfg.whitelist)
            or whitelist_match(ctext, cfg.whitelist) or whitelist_contains(ctext, cfg.whitelist)
        ):
            continue
        ct = label_ctexts.setdefault(label, {})
        if ctext not in ct or len(orig_text) > len(ct[ctext]):
            ct[ctext] = orig_text
        collected_spacy.append((orig_text, label, orig_start, orig_end, "spacy"))

    if cfg.for_spacy_label("PER") != "none":
        covered_spans = regex_spans + [(s, e) for _, _, s, e, _ in collected_spacy]
        for cand_start, cand_end, cand_text in find_lexicon_person_candidates(text):
            if any(cs < cand_end and ce > cand_start for cs, ce in covered_spans):
                continue
            ctext = canonical_text(cand_text, "PER")
            norm_text = cand_text.strip().lower()
            if (
                whitelist_match(norm_text, cfg.whitelist) or whitelist_contains(norm_text, cfg.whitelist)
                or whitelist_match(ctext, cfg.whitelist) or whitelist_contains(ctext, cfg.whitelist)
            ):
                continue
            ct = label_ctexts.setdefault("PER", {})
            if ctext not in ct or len(cand_text) > len(ct[ctext]):
                ct[ctext] = cand_text
            collected_spacy.append((cand_text, "PER", cand_start, cand_end, "lexicon"))

    return collected_spacy, label_ctexts


# ── replacement assignment per coreference cluster ────────────────────────────

def _assign_ne_replacements(
    collected_spacy: list[tuple[str, str, int, int]],
    cluster_map: dict[tuple[str, str], str],
    label_ctexts: dict[str, dict[str, str]],
    config: AnonymizationConfig,
    text: str,
    gen_order: dict[str, list[tuple[str, tuple]]],
    used_rand: dict[str, set[str]],
) -> tuple[dict[tuple, tuple], list[tuple[int, int, tuple]]]:
    """Assigns one replacement per cluster (not per mention). Appends to gen_order in place."""
    s_counters: dict[str, int] = {}
    s_entity_map: dict[tuple, tuple] = {}
    s_span_list: list[tuple[int, int, tuple]] = []

    for orig_text, label, orig_start, orig_end in collected_spacy:
        method = config.for_spacy_label(label)
        ctext = canonical_text(orig_text, label)
        rep_ctext = cluster_map.get((ctext, label), ctext)
        key = (rep_ctext, label)
        if key not in s_entity_map:
            s_counters[label] = s_counters.get(label, 0) + 1
            gen_orig = label_ctexts[label].get(rep_ctext, orig_text)
            gender_hint = None
            if label == "PER" and method == "rand":
                gender_hint = infer_person_gender(gen_orig, text[:orig_start])
            if method == "rand":
                repl, gender, _ = make_unique_rand_replacement(
                    gen_orig, label, s_counters[label], gender_hint, False, used_rand.setdefault(label, set())
                )
            else:
                repl, gender, _ = make_ne_replacement(
                    gen_orig, label, s_counters[label], method, gender_hint
                )
            if method == "gen":
                gen_order.setdefault(repl, []).append(("s", key))
            s_entity_map[key] = (repl, gender)
        s_span_list.append((orig_start, orig_end, key))

    return s_entity_map, s_span_list


def _assign_ne_replacements_detect(
    collected_spacy: list[tuple[str, str, int, int, str]],
    cluster_map: dict[tuple[str, str], str],
    label_ctexts: dict[str, dict[str, str]],
    cfg: AnonymizationConfig,
    text: str,
    gen_order: dict[str, list[tuple[str, tuple]]],
    local_only: bool,
    used_rand: dict[str, set[str]],
) -> tuple[dict[tuple, tuple], dict[tuple, tuple], list[tuple[int, int, tuple, str, str]]]:
    """Same as _assign_ne_replacements, but also tracks (had_local, method) per key."""
    s_counters: dict[str, int] = {}
    s_entity_map: dict[tuple, tuple] = {}
    s_local_map: dict[tuple, tuple] = {}
    s_span_list: list[tuple[int, int, tuple, str, str]] = []

    for orig_text, label, orig_start, orig_end, origin in collected_spacy:
        method = cfg.for_spacy_label(label)
        ctext = canonical_text(orig_text, label)
        rep_ctext = cluster_map.get((ctext, label), ctext)
        key = (rep_ctext, label)
        if key not in s_entity_map:
            s_counters[label] = s_counters.get(label, 0) + 1
            gen_orig = label_ctexts[label].get(rep_ctext, orig_text)
            gender_hint = None
            if label == "PER" and method == "rand":
                gender_hint = infer_person_gender(gen_orig, text[:orig_start])
            if method == "rand":
                repl, gender, had_local = make_unique_rand_replacement(
                    gen_orig, label, s_counters[label], gender_hint, local_only, used_rand.setdefault(label, set())
                )
            else:
                repl, gender, had_local = make_ne_replacement(
                    gen_orig, label, s_counters[label], method, gender_hint, local_only=local_only
                )
            if method == "gen":
                gen_order.setdefault(repl, []).append(("s", key))
            s_entity_map[key] = (repl, gender)
            s_local_map[key] = (had_local, method)
        s_span_list.append((orig_start, orig_end, key, label, origin))

    return s_entity_map, s_local_map, s_span_list


# ── spine ──────────────────────────────────────────────────────────────────

def _run_pipeline(
    text: str, config: AnonymizationConfig,
) -> tuple[str, list[tuple[int, int, str, str | None, int | None, str | None]]]:
    """
    Runs both spacy and regex passes, merges spans, applies replacements. Returns (result, kept_spans)
    """
    _TRANSIENT_GENDER.clear()

    gen_order: dict[str, list[tuple[str, tuple]]] = {}
    used_rand: dict[str, set[str]] = {}
    r_span_list, r_entity_map = _collect_regex_entities(text, config, gen_order)
    regex_spans = [(s, e) for s, e, _ in r_span_list]

    collected_spacy, label_ctexts = _collect_spacy_and_lexicon_entities(text, config, regex_spans)
    cluster_map = cluster_all_labels(label_ctexts)

    s_entity_map, s_span_list = _assign_ne_replacements(
        collected_spacy, cluster_map, label_ctexts, config, text, gen_order, used_rand
    )

    number_duplicate_gen_replacements(gen_order, r_entity_map, s_entity_map)

    plan_regex = [(s, e, *r_entity_map[k]) for s, e, k in r_span_list]
    plan_spacy = [(s, e, *s_entity_map[k]) for s, e, k in s_span_list]
    kept = merge_and_resolve_overlaps(plan_regex + plan_spacy)

    return apply_with_article_fix(text, kept)


def detect_entities(
    text: str,
    config: AnonymizationConfig | None = None,
    *,
    local_only: bool = False,
) -> list[dict]:
    """
    Run detection passes without applying replacements.
    Returns list of {id, start, end, original, replacement, label, source, method,
    local_match, core_start, core_replacement}.
    start/end/original/replacement are article-agreement-aware (Section 4.7): a
    replacement whose gender differs from the original can pull a preceding
    article/preposition into the span so the reported text matches what actually
    gets spliced into the output. core_start/core_replacement are the same span
    and replacement *before* that article rewrite is folded in -- the values
    Consistent Replacement (Section 4.5.2) actually guarantees are identical
    across every mention of the same cluster, and the tight span the pipeline
    itself identified as the entity. Use start/replacement to reconstruct real
    output text; use core_start/core_replacement for span-containment or
    replacement-identity checks (e.g. against a tightly-annotated gold span).
    local_only=True skips live Wikidata — replacements may be static fallbacks.
    """
    cfg = resolve_config(config)
    _TRANSIENT_GENDER.clear()

    gen_order: dict[str, list[tuple[str, tuple]]] = {}
    used_rand: dict[str, set[str]] = {}
    r_span_list, r_entity_map = _collect_regex_entities_detect(text, cfg, gen_order)
    regex_spans = [(s, e) for s, e, _, _ in r_span_list]

    collected_spacy, label_ctexts = _collect_spacy_and_lexicon_entities_detect(text, cfg, regex_spans)
    cluster_map = cluster_all_labels(label_ctexts)

    s_entity_map, s_local_map, s_span_list = _assign_ne_replacements_detect(
        collected_spacy, cluster_map, label_ctexts, cfg, text, gen_order, local_only, used_rand
    )

    number_duplicate_gen_replacements(gen_order, r_entity_map, s_entity_map)

    plan_regex = [(s, e, *r_entity_map[k], nl, "regex") for s, e, k, nl in r_span_list]
    plan_spacy = [(s, e, *s_entity_map[k], nl, origin) for s, e, k, nl, origin in s_span_list]

    # lookup tables for local_match and method per span position
    span_local: dict[tuple[int, int], bool] = {}
    span_method: dict[tuple[int, int], str] = {}
    for s, e, k, nl in r_span_list:
        span_local[(s, e)] = True
        span_method[(s, e)] = cfg.regex
    for s, e, k, nl, origin in s_span_list:
        had_local, method = s_local_map.get(k, (True, "tag"))
        span_local[(s, e)] = had_local
        span_method[(s, e)] = method

    kept = merge_and_resolve_overlaps(plan_regex + plan_spacy)

    # Simulate article-fixing pass to compute article-aware spans
    art_info = simulate_article_fix(text, kept)

    entities: list[dict] = []
    for i, (start, end, repl, gender, label, source) in enumerate(kept):
        art = art_info.get(i)
        if art:
            art_start, art_repl = art
            eff_start = art_start
            eff_original = text[art_start:end]
            eff_replacement = art_repl + repl
        else:
            eff_start = start
            eff_original = text[start:end]
            eff_replacement = repl

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


# ── public API ────────────────────────────────────────────────────────────


def pipeline_with_spans(
    text:   str,
    config: AnonymizationConfig | None = None,
) -> tuple[str, list[AnonSpan]]:
    """
    Full anonymization pipeline but also returns char-offset spans of every anonymized entity
    """
    result, kept = _run_pipeline(text, resolve_config(config))
    spans = []
    for start, end, repl, gender, art_start, art_repl in kept:
        if art_start is not None:
            spans.append(AnonSpan(start=art_start, end=end, original=text[art_start:end], replacement=art_repl + repl))
        else:
            spans.append(AnonSpan(start=start, end=end, original=text[start:end], replacement=repl))
    return result, spans
