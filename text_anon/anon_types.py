from dataclasses import dataclass, field
from difflib import SequenceMatcher

VALID_METHODS = frozenset({"sup", "tag", "rand", "gen", "none"})


@dataclass
class AnonymizationConfig:
    # to be able to treat each entity type independently
    regex: str = "tag"
    per:   str = "rand"
    loc:   str = "gen"
    org:   str = "gen"
    misc:  str = "tag"
    # regex labels listed here are left untouched regardless of the regex method
    regex_skip_labels: frozenset[str] = field(default_factory=frozenset)
    # specific words/phrases that should never be anonymized (case-insensitive)
    whitelist: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        for attr in ("regex", "per", "loc", "org", "misc"):
            val = getattr(self, attr)
            if val not in VALID_METHODS:
                raise ValueError(
                    f"Inválido '{val}' para '{attr}'. "
                    f"Escolhe de: {', '.join(sorted(VALID_METHODS))}"
                )
        self.regex_skip_labels = frozenset(self.regex_skip_labels)
        self.whitelist = frozenset(w.strip().lower() for w in self.whitelist if w.strip())

    def for_spacy_label(self, label: str) -> str:
        return {"PER": self.per, "LOC": self.loc, "ORG": self.org, "MISC": self.misc}.get(
            label, self.misc
        )


@dataclass
class AnonSpan:
    # to know what changed and where
    start: int
    end: int
    original: str
    replacement: str


def resolve_config(config: AnonymizationConfig | None) -> AnonymizationConfig:
    """
    Define anonymization config
    """
    return config if config is not None else AnonymizationConfig()


_WHITELIST_FUZZY_MIN_LEN = 4
_WHITELIST_FUZZY_RATIO   = 0.80

# always-on exclusions, invisible to AnonymizationConfig.whitelist -- these are
# structured-identifier field labels that spaCy's NER occasionally mistags as
# MISC on their own (e.g. the bare word "NIF" or "email"), separate from the
# actual identifier value, which regex detection already handles correctly
_BUILTIN_WHITELIST: frozenset[str] = frozenset({"nif", "niss", "nib", "email", "enderço", "contacto", "morada", "querido", "matrícula", "iban", "cc"})


def whitelist_match(text: str, whitelist: frozenset) -> bool:
    whitelist = whitelist | _BUILTIN_WHITELIST
    if text in whitelist:
        return True
    for entry in whitelist:
        if len(entry) < _WHITELIST_FUZZY_MIN_LEN:
            continue
        if SequenceMatcher(None, text, entry).ratio() >= _WHITELIST_FUZZY_RATIO:
            return True
    return False


def whitelist_contains(text: str, whitelist: frozenset) -> bool:
    """
    True if a whitelist entry is embedded in `text` on word boundaries, or
    `text` is embedded in a (multi-word) whitelist entry -- catches a
    detected span that grabbed extra surrounding words alongside a
    whitelisted term (e.g. "Receitei Crotax" containing "crotax"), or a
    detected span that's just one word of a longer whitelisted phrase (e.g.
    "Ringer" inside "lactato de ringer"). whitelist_match()'s whole-string
    ratio can't catch either shape, since the two strings are deliberately
    different lengths, not near-duplicates. Word-boundary based (splitting on
    whitespace) rather than a raw substring check, so a short entry can't
    spuriously match inside an unrelated longer word.
    """
    words = text.split()
    if not words:
        return False
    for entry in whitelist | _BUILTIN_WHITELIST:
        entry_words = entry.split()
        if not entry_words:
            continue
        if len(entry_words) <= len(words):
            shorter, longer = entry_words, words
        else:
            shorter, longer = words, entry_words
        n = len(shorter)
        if any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1)):
            return True
    return False
