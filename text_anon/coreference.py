import re

from .detection import TITLE_PATTERN

_CORP_SUFFIXES = re.compile(
    r"\s+(S\.?A\.?|Lda\.?|Ltda\.?|Unipessoal\s+Lda\.?|SARL|SGPS|SRL|S\.?R\.?L\.?)\s*$",
    re.IGNORECASE,
)


def canonical_text(text: str, label: str) -> str:
    """Strip titles and corporate suffixes for coreference key normalisation."""
    t = TITLE_PATTERN.sub("", text).strip().lower()
    if label == "ORG":
        t = _CORP_SUFFIXES.sub("", t).strip()
    return re.sub(r"\s+", " ", t)


def cluster_entities(canonical_texts: list[str], label: str) -> dict[str, str]:
    """
    Group canonical entity texts that refer to the same object.
    Returns {canonical_text: cluster_representative} for every input text.
    Rules applied:
      PER  — partial token match, only when unambiguous
      ORG/LOC/MISC — acronym match, only when unambiguous
    """
    if len(canonical_texts) <= 1:
        return {t: t for t in canonical_texts}

    parent = {t: t for t in canonical_texts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # prefer the longer (more complete) form as cluster root
            if len(ra) >= len(rb):
                parent[rb] = ra
            else:
                parent[ra] = rb

    texts = list(canonical_texts)

    if label == "PER":
        token_sets = {t: set(t.split()) for t in texts}
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                ta, tb = token_sets[a], token_sets[b]
                if ta <= tb or tb <= ta:
                    smaller = ta if len(ta) <= len(tb) else tb
                    # ambiguous only if a third entity (outside this pair) shares the tokens
                    ambiguous = any(
                        smaller <= token_sets[c] for c in texts if c != a and c != b
                    )
                    if not ambiguous:
                        union(a, b)

    elif label in ("ORG", "LOC", "MISC"):
        _STOP = {"de", "da", "do", "dos", "das", "e", "o", "a", "os", "as",
                 "em", "no", "na", "nos", "nas", "por", "para", "com", "the", "of"}

        def _initials(t: str) -> str:
            return "".join(w[0] for w in t.split() if w and w not in _STOP)

        ini_map = {t: _initials(t) for t in texts}
        ini_to_full: dict[str, list[str]] = {}
        for t, ini in ini_map.items():
            ini_to_full.setdefault(ini, []).append(t)
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                ini_a, ini_b = ini_map[a], ini_map[b]
                # a is an acronym of b (unambiguous: only one full form)
                if a == ini_b and len(ini_to_full.get(ini_b, [])) == 1:
                    union(a, b)
                elif b == ini_a and len(ini_to_full.get(ini_a, [])) == 1:
                    union(a, b)

    clusters: dict[str, list[str]] = {}
    for t in texts:
        clusters.setdefault(find(t), []).append(t)

    result: dict[str, str] = {}
    for members in clusters.values():
        rep = max(members, key=len)
        for m in members:
            result[m] = rep
    return result


def cluster_all_labels(label_ctexts: dict[str, dict[str, str]]) -> dict[tuple[str, str], str]:
    """
    Clusters canonical texts per label, across all labels present.
    Returns {(canonical_text, label): cluster_representative}.
    """
    cluster_map: dict[tuple[str, str], str] = {}
    for label, ct in label_ctexts.items():
        for ctext, rep in cluster_entities(list(ct.keys()), label).items():
            cluster_map[(ctext, label)] = rep
    return cluster_map
