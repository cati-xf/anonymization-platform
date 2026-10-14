import re
import random

from ..detection.lexicon_search import NAMES_M, NAMES_F, SURNAMES, NAMES_M_FULL_SET, NAMES_F_FULL_SET

_MASC_ARTICLES = frozenset({
    "o", "um", "do", "ao", "no", "pelo", "dum", "num",
    "os", "dos", "aos", "nos", "pelos",
})
_FEM_ARTICLES = frozenset({
    "a", "uma", "da", "à", "na", "pela", "duma", "numa",
    "as", "das", "às", "nas", "pelas",
})

SUBSTITUTION_POOL_WITH_GENDER: dict[str, list[tuple[str, str | None]]] = {
    "PER": [
        ("Miguel", "m"), ("Ana", "f"), ("João", "m"), ("Sofia", "f"),
        ("Carlos", "m"), ("Beatriz", "f"), ("Rui", "m"), ("Inês", "f"),
        ("Pedro", "m"), ("Mariana", "f"), ("Tiago", "m"), ("Catarina", "f"),
        ("Luís", "m"), ("Filipa", "f"), ("André", "m"), ("Teresa", "f"),
    ],
    "LOC": [
        ("Coimbra", "f"), ("Porto", "m"), ("Évora", "f"), ("Braga", "f"),
        ("Setúbal", "f"), ("Viseu", "m"), ("Faro", "m"), ("Aveiro", "m"),
        ("Guimarães", "f"), ("Leiria", "f"), ("Beja", "f"),
        ("Viana do Castelo", "f"), ("Vila Real", "f"), ("Bragança", "f"),
    ],
    "ORG": [
        ("Empresa Nacional", "f"), ("Grupo Atlântico", "m"),
        ("Instituto Central", "m"), ("Fundação Portuguesa", "f"),
        ("Cooperativa Sul", "f"), ("Centro de Investigação", "m"),
        ("Serviços Integrados", "m"),
    ],
    "MISC": [("termo A", "m"), ("termo B", "m"), ("termo C", "m")],
}


def infer_person_gender(name: str, preceding_text: str) -> str | None:
    """
    Infer gender of a person by checking lexicons and as fallback article
    """
    tokens = name.strip().split()
    first_token = tokens[0].lower() if tokens else ""
    if first_token in NAMES_M_FULL_SET:
        return "m"
    if first_token in NAMES_F_FULL_SET:
        return "f"

    words = re.findall(r"\S+", preceding_text)
    if words:
        last_word = words[-1].lower().rstrip(",;:")
        if last_word in _MASC_ARTICLES:
            return "m"
        if last_word in _FEM_ARTICLES:
            return "f"

    return None


def pick_random_person(gender: str | None = None) -> tuple[str, str]:
    """
    Use retrieved lexicons to select random substitution for names
    """
    if NAMES_M and NAMES_F and SURNAMES:
        if gender == "m" or (gender is None and random.random() < 0.5):
            first, g = random.choice(NAMES_M), "m"
        else:
            first, g = random.choice(NAMES_F), "f"
        last = random.choice(SURNAMES)
        return f"{first} {last}", g

    # fallback in case of no lexicon
    pool = SUBSTITUTION_POOL_WITH_GENDER["PER"]
    if gender in ("m", "f"):
        pool = [entry for entry in pool if entry[1] == gender]
    chosen, g = random.choice(pool)
    return chosen, g
