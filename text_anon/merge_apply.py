from .article import fix_article, ARTICLE_PATTERN, BARE_PREP_PATTERN, _TRANSIENT_GENDER


def merge_and_resolve_overlaps(spans: list[tuple]) -> list[tuple]:
    """
    Sorts spans by (start, -length) and drops any span overlapping an
    already-kept span (longer span wins on ties). Works on any tuple whose
    first two elements are (start, end).
    """
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    kept: list[tuple] = []
    last_end = -1
    for span in spans:
        if span[0] >= last_end:
            kept.append(span)
            last_end = span[1]
    return kept


def apply_with_article_fix(
    text: str,
    kept: list[tuple[int, int, str, str | None]],
) -> tuple[str, list[tuple[int, int, str, str | None, int | None, str | None]]]:
    """
    Applies (start, end, replacement, gender) spans right-to-left, fixing the
    preceding article/preposition for gender agreement before each splice.
    Returns (result_text, kept_spans_with_article_info).
    """
    result = text

    new_kept: list[tuple[int, int, str, str | None, int | None, str | None]] = []
    for start, end, repl, gender in reversed(kept):
        if gender:
            _TRANSIENT_GENDER[repl] = gender
        preceding = result[:start]
        fixed_preceding = fix_article(preceding, repl)
        art_start: int | None = None
        art_repl: str | None = None
        if fixed_preceding != preceding:
            m = ARTICLE_PATTERN.search(preceding) or BARE_PREP_PATTERN.search(preceding)
            if m:
                art_start = m.start(1)
                art_repl = fixed_preceding[m.start(1):]
        result = fixed_preceding + repl + result[end:]
        new_kept.append((start, end, repl, gender, art_start, art_repl))

    new_kept.reverse()
    return result, new_kept


def simulate_article_fix(
    text: str,
    kept: list[tuple],
) -> dict[int, tuple[int, str]]:
    """
    Simulates the article-fixing pass without splicing the actual result text
    (used by detect_entities, which reports spans/offsets rather than applying
    them immediately). kept items are (start, end, repl, gender, label, source).
    Returns {span_index: (art_start, fixed_preceding_from_art_start)}.
    """
    sim_text = text
    art_info: dict[int, tuple[int, str]] = {}
    for i, (start, end, repl, gender, label, source) in enumerate(reversed(kept)):
        idx = len(kept) - 1 - i
        if gender:
            _TRANSIENT_GENDER[repl] = gender
        preceding = sim_text[:start]
        fixed_preceding = fix_article(preceding, repl)
        if fixed_preceding != preceding:
            m_art = ARTICLE_PATTERN.search(preceding) or BARE_PREP_PATTERN.search(preceding)
            if m_art:
                art_info[idx] = (m_art.start(1), fixed_preceding[m_art.start(1):])
        sim_text = fixed_preceding + repl + sim_text[end:]
    return art_info


def apply_entities(text: str, entities: list[dict]) -> str:
    """Apply entity substitutions to text in reverse order. Empty replacements are skipped."""
    result = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        repl = ent.get("replacement") or ""
        if repl:
            result = result[:ent["start"]] + repl + result[ent["end"]:]
    return result
