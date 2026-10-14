import re

import spacy

nlp = spacy.load("pt_core_news_lg")


TITLE_PATTERN = re.compile(
    r"\b(Dr\.?a?|Dra\.?|Sr\.?a?|Sra\.?|Prof\.?a?|Profa\.?|Eng\.?a?|Enga\.?|"
    r"Doutor|Doutora|Senhor|Senhora|Professor|Professora|Engenheiro|Engenheira|"
    r"Dom|Dona|D\.)\s+",
    re.IGNORECASE,
)
_TITLE_PLACEHOLDER = "pessoa"


def mask_titles(text: str) -> tuple[str, list[tuple[str, int]]]:
    """
    Replace each title with a fixed placeholder and return (original, start pos)
    """
    replacements: list[tuple[str, int]] = []
    offset = 0
    result = text

    for m in TITLE_PATTERN.finditer(text):
        start = m.start() + offset
        original = m.group(0)
        placeholder = _TITLE_PLACEHOLDER + " "

        result = result[:start] + placeholder + result[start + len(original):]
        replacements.append((original, start))
        offset += len(placeholder) - len(original)

    return result, replacements


def build_masked_to_orig(
    original: str,
    replacements: list[tuple[str, int]],
) -> list[int]:
    """
    Returns offset_map where offset_map[i] = position in original for
    position i in the masked string
    """
    placeholder_len = len(_TITLE_PLACEHOLDER) + 1
    intervals: list[tuple[int, int, int, int]] = []

    o_cursor = 0
    for orig_title, m_start in replacements:
        idx = original.find(orig_title, o_cursor)
        if idx == -1:
            continue
        intervals.append((m_start, m_start + placeholder_len, idx, idx + len(orig_title)))
        o_cursor = idx + len(orig_title)

    offset_map: list[int] = []
    m_pos = 0
    o_pos = 0
    iv = 0
    masked_len = len(original) + sum(
        placeholder_len - len(orig) for orig, _ in replacements
    )
    while m_pos <= masked_len:
        if iv < len(intervals) and m_pos == intervals[iv][0]:
            m_s, m_e, o_s, o_e = intervals[iv]
            offset_map.extend([o_s] * (m_e - m_s))
            m_pos, o_pos = m_e, o_e
            iv += 1
        else:
            offset_map.append(o_pos)
            m_pos += 1
            o_pos += 1
    return offset_map
