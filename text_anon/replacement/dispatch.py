import re
import random

from ..detection import REGEX_READABLE
from ..article import GENDER, _generalize_datetime, _generalize_quantity
from .wikidata import lookup_gen, wikidata_substitutes, _gender_from_label
from .random_person import pick_random_person, SUBSTITUTION_POOL_WITH_GENDER


def generalize_ne(text: str, label: str, local_only: bool = False) -> tuple[str, str | None, bool]:
    if label == "PER":
        return "pessoa anónima", GENDER["pessoa"], True

    pt_label, gender, had_local = lookup_gen(text, label, local_only=local_only)
    if pt_label:
        if not gender:
            gender = _gender_from_label(pt_label)
        suffix = "anónima" if gender == "f" else "anónimo"
        return f"{pt_label} {suffix}", gender, had_local

    if label == "ORG":
        fallback = "organização anónima"
    elif label == "LOC":
        fallback = "localidade anónima"
    else:
        fallback = "termo anónimo"
    return fallback, GENDER.get(fallback), False


def make_regex_replacement(
    label: str,
    original: str,
    idx: int,
    method: str,
) -> tuple[str, str | None]:
    """
    Returns the replacement (and associated gender if needed) for regex entities.
    No distinction between random substitution and generalization, except for
    DATETIME/QUANTITY/IDADE, whose spans are just the bare value (no unit words
    baked in) so the replacement has to stay a bare value too.
    """
    if method == "sup":
        return "XXXXXX", None

    if method == "tag":
        return f"[{label}_{idx}]", None

    # if datetime, quantity, or idade, check which type, else regex_readable
    if method in ("rand", "gen"):
        if label == "DATETIME":
            base   = _generalize_datetime(original)
            gender = "m" if re.fullmatch(r"\d{4}", base) else GENDER.get(base)
            return base, gender

        if label == "QUANTITY":
            base = _generalize_quantity(original)
            return base, GENDER.get(base)

        if label == "IDADE" and original.isdigit():
            n = int(original)
            if method == "gen":
                return str((n // 10) * 10), "m"
            delta = random.choice([d for d in range(-10, 11) if d != 0])
            return str(max(1, min(99, n + delta))), "m"

        base = REGEX_READABLE.get(label, label.lower())
        return base, GENDER.get(base)

    return f"[{label}_{idx}]", None


def make_ne_replacement(
    original: str,
    label: str,
    idx: int,
    method: str,
    gender_hint: str | None = None,
    local_only: bool = False,
) -> tuple[str, str | None, bool]:
    """
    Returns (replacement, gender, had_local) for PER, LOC, ORG, MISC.
    had_local=True when no live Wikidata call was needed.
    """
    if method == "sup":
        return "XXXXXX", None, True

    if method == "tag":
        return f"[{label}_{idx}]", None, True

    if method == "rand":
        if label == "PER":
            repl, gender = pick_random_person(gender=gender_hint)
            return repl, gender, True

        if label in ("LOC", "ORG", "MISC"):
            candidates, had_local = wikidata_substitutes(original, label, local_only=local_only)
            if candidates:
                chosen, gender = random.choice(candidates)
                if gender is None:
                    gender = _gender_from_label(chosen)
                return chosen, gender, had_local

        pool = SUBSTITUTION_POOL_WITH_GENDER.get(label, SUBSTITUTION_POOL_WITH_GENDER["MISC"])
        chosen, gender = random.choice(pool)
        return chosen, gender, False

    if method == "gen":
        return generalize_ne(original, label, local_only=local_only)

    return f"[{label}_{idx}]", None, True


def number_duplicate_gen_replacements(
    gen_order: dict[str, list[tuple[str, tuple]]],
    r_entity_map: dict[tuple, tuple],
    s_entity_map: dict[tuple, tuple],
) -> None:
    """
    Numbers gen replacements only when the same base label covers multiple distinct entities.
    Mutates r_entity_map / s_entity_map in place.
    """
    for base_label, entries in gen_order.items():
        if len(entries) > 1:
            for idx, (source, key) in enumerate(entries, 1):
                emap = r_entity_map if source == "r" else s_entity_map
                repl, gender = emap[key]
                emap[key] = (f"{repl} {idx}", gender)


def make_unique_rand_replacement(
    original: str,
    label: str,
    idx: int,
    gender_hint: str | None,
    local_only: bool,
    used: set[str],
    max_attempts: int = 6,
) -> tuple[str, str | None, bool]:
    """
    Like make_ne_replacement(method="rand"), but avoids handing two different
    real-world entities the identical replacement text within one document --
    a rand draw is randomized (pick_random_person / wikidata_substitutes +
    random.choice), so two different people/places can otherwise legitimately
    collide by chance (the hard-coded fallback pools in random_person.py are
    only 11-16 entries). Unlike gen, which is deliberately generic and numbers
    every collision up front (number_duplicate_gen_replacements), a rand
    replacement is meant to read as a specific real name/place, so retrying
    for a *different* real value first keeps the text natural; only once the
    retry budget is exhausted does this fall back to the same numeric-suffix
    disambiguation gen already uses.

    Retries are cheap: wikidata_substitutes()'s candidate list is cached per
    (term, label) in substitution_cache after the first lookup, so a retry for
    the same entity just re-randomizes over an already-fetched list, no repeat
    network calls; pick_random_person() is a pure in-memory lexicon draw.

    `used` is mutated in place -- the set of replacements already handed out
    for this label in this document (caller passes one shared set per label,
    reused across every cluster of that label in the same detect_entities()/
    _run_pipeline() call).
    """
    repl = gender = had_local = None
    for _ in range(max_attempts):
        repl, gender, had_local = make_ne_replacement(original, label, idx, "rand", gender_hint, local_only=local_only)
        if repl not in used:
            used.add(repl)
            return repl, gender, had_local

    suffix = 2
    numbered = f"{repl} {suffix}"
    while numbered in used:
        suffix += 1
        numbered = f"{repl} {suffix}"
    used.add(numbered)
    return numbered, gender, had_local


def lookup_live(term: str, ner_label: str, method: str) -> str | None:
    """
    Run a live Wikidata lookup for a single entity and return the replacement string.
    Bypasses the local_only constraint — always reaches out to Wikidata if needed.
    Returns None when nothing is found even with live query.
    """
    if method == "gen":
        pt_label, gender, _ = lookup_gen(term, ner_label, local_only=False)
        if pt_label:
            gender = gender or _gender_from_label(pt_label)
            suffix = "anónima" if gender == "f" else "anónimo"
            return f"{pt_label} {suffix}"
    elif method == "rand" and ner_label in ("LOC", "ORG", "MISC"):
        candidates, _ = wikidata_substitutes(term, ner_label, local_only=False)
        if candidates:
            chosen, _ = random.choice(candidates)
            return chosen
    return None
