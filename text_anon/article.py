import re

from .detection import REGEX_GENDER as _REGEX_GENDER

GENDER: dict[str, str] = {
    **_REGEX_GENDER,
    "hora":                 "f",
    "data":                 "f",
    "valor monetário":      "m",
    "quantidade":           "f",
    "percentagem":          "f",
    "pessoa":               "f",
    "organização":          "f",
    "entidade":             "f",
    "localidade":           "f",
    "cidade":               "f",
    "termo":                "m",
}

# gender values for wikidata words (cleared per pipeline run)
_TRANSIENT_GENDER: dict[str, str] = {}

ARTICLES = {
    ("o",   "f"): "a",    ("o",   "m"): "o",
    ("a",   "m"): "o",    ("a",   "f"): "a",
    ("um",  "f"): "uma",  ("um",  "m"): "um",
    ("uma", "m"): "um",   ("uma", "f"): "uma",
    ("no",  "f"): "na",   ("no",  "m"): "no",
    ("na",  "m"): "no",   ("na",  "f"): "na",
    ("num", "f"): "numa", ("num", "m"): "num",
    ("numa","m"): "num",  ("numa","f"): "numa",
    ("do",  "f"): "da",   ("do",  "m"): "do",
    ("da",  "m"): "do",   ("da",  "f"): "da",
    ("dum", "f"): "duma", ("dum", "m"): "dum",
    ("duma","m"): "dum",  ("duma","f"): "duma",
    ("pelo","f"): "pela", ("pelo","m"): "pelo",
    ("pela","m"): "pelo", ("pela","f"): "pela",
}

ARTICLE_PATTERN = re.compile(
    r'\b(pelo|pela|duma|dum|numa|num|uma|um|do|da|no|na|ao|à|os|as|uns|umas'
    r'|nos|nas|dos|das|aos|às|pelos|pelas|o|a)\s+$',
    re.IGNORECASE,
)

BARE_PREP_PATTERN = re.compile(
    r'\b(em|de|a|por)\s+$',
    re.IGNORECASE,
)

BARE_PREP_CONTRACTION = {
    ("em", "m"): "no",   ("em", "f"): "na",
    ("de", "m"): "do",   ("de", "f"): "da",
    ("a",  "m"): "ao",   ("a",  "f"): "à",
    ("por","m"): "pelo", ("por","f"): "pela",
}

# any article/preposition that can precede a date, so a bare year left
# behind by _generalize_datetime can be forced to take "em" instead of
# going through gender agreement (e.g. "a 7 de novembro de 2003" -> "em 2003",
# not "o 2003")
_PREP_OR_ARTICLE_PATTERN = re.compile(
    r'\b(pelo|pela|duma|dum|numa|num|uma|um|do|da|no|na|ao|à|os|as|uns|umas'
    r'|nos|nas|dos|das|aos|às|pelos|pelas|o|a|em|de|por)\s+$',
    re.IGNORECASE,
)

_BARE_YEAR_RE = re.compile(r"\A\d{4}\Z")

_YEAR_RE     = re.compile(r"\b(\d{4})\b")
_TIME_RE     = re.compile(r"\b\d{1,2}:\d{2}")
_MONETARY_RE = re.compile(r"€|euros?|dólares?|libras?|dollars?|pounds?", re.IGNORECASE)
_PERCENT_RE  = re.compile(r"%|por\s+cento",                              re.IGNORECASE)


def _get_gender(replacement: str) -> str | None:
    return (
        GENDER.get(replacement)
        or GENDER.get(replacement.lower())
        or _TRANSIENT_GENDER.get(replacement)
        or _TRANSIENT_GENDER.get(replacement.lower())
    )


def fix_article(text: str, replacement: str) -> str:
    """
    Looks at text immediately before and makes the gender match
    """
    if _BARE_YEAR_RE.match(replacement):
        m = _PREP_OR_ARTICLE_PATTERN.search(text)
        if m:
            new_prep = "Em" if m.group(1)[0].isupper() else "em"
            return text[: m.start(1)] + new_prep + text[m.end(1):]
        return text

    gender = _get_gender(replacement)
    if not gender:
        return text

    m = ARTICLE_PATTERN.search(text)
    if m:
        article     = m.group(1).lower()
        new_article = ARTICLES.get((article, gender))
        if new_article:
            if m.group(1)[0].isupper():
                new_article = new_article.capitalize()
            return text[: m.start(1)] + new_article + text[m.end(1):]

    m = BARE_PREP_PATTERN.search(text)
    if m:
        prep        = m.group(1).lower()
        contraction = BARE_PREP_CONTRACTION.get((prep, gender))
        if contraction:
            if m.group(1)[0].isupper():
                contraction = contraction.capitalize()
            return text[: m.start(1)] + contraction + text[m.end(1):]

    return text


def _generalize_datetime(original: str) -> str:
    if _TIME_RE.match(original.strip()) and not _YEAR_RE.search(original):
        return "hora anónima"
    m = _YEAR_RE.search(original)
    if m:
        return m.group(1)
    return "data anónima"


def _generalize_quantity(original: str) -> str:
    if _MONETARY_RE.search(original):
        return "valor monetário anónimo"
    if _PERCENT_RE.search(original):
        return "percentagem anónima"
    return "quantidade anónima"
