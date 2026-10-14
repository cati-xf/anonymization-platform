import os
import re

# package root (text_anon/), where the lexicons/ CSV directory lives
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_lexicon(filename: str) -> list[str]:
    path = os.path.join(_HERE, "lexicons", filename)
    try:
        with open(path, encoding="utf-8-sig") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


NAMES_M:      list[str]      = _load_lexicon("nomes_masculinos.csv")  # from official list irn
NAMES_F:      list[str]      = _load_lexicon("nomes_femininos.csv")   # from official list irn
_NAMES_M_SET: frozenset[str] = frozenset(n.lower() for n in NAMES_M)  # for lookup
_NAMES_F_SET: frozenset[str] = frozenset(n.lower() for n in NAMES_F)
SURNAMES:     list[str]      = _load_lexicon("apelidos.csv")          # from wikipedia

_NAMES_M_COMPLETO: list[str] = _load_lexicon("nomes_masculinos_completo.csv")
_NAMES_F_COMPLETO: list[str] = _load_lexicon("nomes_femininos_completo.csv")

# broader dictionaries used as a detection safety net (may include non-PT/contaminated entries)
NAMES_M_FULL_SET:     frozenset[str] = _NAMES_M_SET | frozenset(n.lower() for n in _NAMES_M_COMPLETO)
NAMES_F_FULL_SET:     frozenset[str] = _NAMES_F_SET | frozenset(n.lower() for n in _NAMES_F_COMPLETO)
_ALL_FIRST_NAMES_SET: frozenset[str] = NAMES_M_FULL_SET | NAMES_F_FULL_SET
_SURNAMES_SET:        frozenset[str] = frozenset(n.lower() for n in SURNAMES)


_WORD_TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def find_lexicon_person_candidates(text: str) -> list[tuple[int, int, str]]:
    """
    Find name-lexicon matches (nomes_*_completo.csv + apelidos.csv) the caller
    can use as a gap-filler for PER entities missed by spaCy. Case-insensitive,
    so ASR-lowercased names are caught too. Adjacent matching tokens (separated
    only by whitespace) are merged into a single candidate span.
    """
    tokens = list(_WORD_TOKEN_PATTERN.finditer(text))
    candidates: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        if tokens[i].group().lower() not in _ALL_FIRST_NAMES_SET and tokens[i].group().lower() not in _SURNAMES_SET:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(tokens)
            and text[tokens[j].end():tokens[j + 1].start()].isspace()
            and (tokens[j + 1].group().lower() in _ALL_FIRST_NAMES_SET or tokens[j + 1].group().lower() in _SURNAMES_SET)
        ):
            j += 1
        start, end = tokens[i].start(), tokens[j].end()
        candidates.append((start, end, text[start:end]))
        i = j + 1
    return candidates
