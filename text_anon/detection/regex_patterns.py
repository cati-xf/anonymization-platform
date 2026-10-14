"""
Portuguese entity classes
  NIF              – Tax Identification Number
  TELEMOVEL        – Telephone number
  CARTAO_CIDADAO   – Citizen Card number (reduced or full)
  CARTAO_BANCARIO  – Credit / debit card 
  IBAN             – International Bank Account Number
  NIB              – Bank identification number 
  NISS             – Social Security Number
  PASSAPORTE       – Passport number
  CARTA_CONDUCAO   – Driving licence
  NUM_UTENTE       – National Health Service patient number
  EMAIL            – E-mail address
  CODIGO_POSTAL    – Postal code
  MATRICULA        – Vehicle registration plate
  DATETIME         – Date / time expressions
  QUANTITY         – Monetary values, percentages, physical quantities
  IDADE            - Age
  NUM_PORTA        - Address door number 

"""

import re
from dataclasses import dataclass
from typing import Callable

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegexMatch:
    # for a single regex span
    label: str
    start: int
    end:   int
    text:  str

    def __len__(self) -> int:
        return self.end - self.start


# ── Checksum / validation helpers ─────────────────────────────────────────────

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _validate_nif(raw: str) -> bool:
    """
        - multiply digit[7] by 2, digit[6] by 3, …, digit[0] by 9
        - sum the products
        - remainder = sum % 11
        - check digit (digit[8]) = 0 if remainder is 0 or 1 
        - else check digit = 11 − remainder
    """
    digits = _digits_only(raw.upper().removeprefix("PT"))
    if len(digits) != 9 or digits[0] == "0":
        return False
    total = sum(int(d) * (9 - i) for i, d in enumerate(digits[:8]))
    remainder = total % 11
    expected = 0 if remainder in (0, 1) else 11 - remainder
    return int(digits[8]) == expected


def _validate_cartao_cidadao(raw: str) -> bool:
    """
    for full citizen card number (12 characters)
        - iterate characters right-to-left
        - convert letters via A=10 … Z=35
        - double every second value 
        - if doubled value >= 10, subtract 9
        - sum all values, valid iff sum % 10 == 0
    """
    LETTER_VAL = {chr(65 + i): 10 + i for i in range(26)}

    s = re.sub(r"[\s-]+", "", raw).upper()
    if len(s) != 12:
        return False
    total = 0
    for idx, ch in enumerate(reversed(s)):
        if ch.isdigit():
            val = int(ch)
        elif ch in LETTER_VAL:
            val = LETTER_VAL[ch]
        else:
            return False
        if idx % 2 == 1:    
            val *= 2
            if val >= 10:
                val -= 9
        total += val
    return total % 10 == 0


def _validate_nib(raw: str) -> bool:
    """
    NIB is valid when the corresponding IBAN is valid
    """
    digits = _digits_only(raw)
    if len(digits) != 21:
        return False
    return _validate_iban("PT50" + digits)


def _validate_iban(raw: str) -> bool:
    """
    ISO 13616
        - move first 4 chars to end
        - convert letters to digits
        - valid iff int % 97 == 1
    """
    s = re.sub(r"\s+", "", raw).upper()
    if len(s) < 15:
        return False
    rearranged = s[4:] + s[:4]
    sum = ""
    for ch in rearranged:
        if ch.isdigit():
            sum += ch
        elif ch.isalpha():
            sum += str(ord(ch) - 55)
        else:
            return False
    return int(sum) % 97 == 1


def _validate_niss(raw: str) -> bool:
    """
        - sum of the first 10 digit products by the weights
        - rest of the division by 10
        - subtract the result from 9
        - valid iff check == digits[10]
    """
    digits = _digits_only(raw)
    if len(digits) != 11:
        return False
    weights = [29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
    total = sum(int(d) * w for d, w in zip(digits[:10], weights))
    expected = (9 - (total % 10)) % 10
    return int(digits[10]) == expected


def _validate_luhn(raw: str) -> bool:
    """
        - iterate digits right-to-left
        - double every second digit, subtract 9 if result >= 10
        - valid iff sum of all digits % 10 == 0
    """
    digits = _digits_only(raw)
    if not digits:
        return False
    total = 0
    for idx, ch in enumerate(reversed(digits)):
        val = int(ch)
        if idx % 2 == 1:
            val *= 2
            if val >= 10:
                val -= 9
        total += val
    return total % 10 == 0


# ── Context ───────────────────────────────────────────────────────────────────

_CONTEXT_WORDS = 15

def _has_context(text: str, start: int, end: int, keywords: list[str]) -> bool:
    """
    return true if any keyword is within CONTEXT_WORDS before/after the match,
    as a whole word -- a substring check would let e.g. "morada" match inside
    "namorada"
    """
    before = " ".join(re.findall(r"\S+", text[:start])[-_CONTEXT_WORDS:])
    after  = " ".join(re.findall(r"\S+", text[end:])[:_CONTEXT_WORDS])
    window = (before + " " + after).lower()
    return any(re.search(rf"\b{re.escape(kw.lower())}\b", window) for kw in keywords)


# ── Simple List Builder ───────────────────────────────────────────────────────

""" 
RegexMatch list builder for simple entities (no extra logic needed)
"""
def _simple(label: str, pattern: re.Pattern[str], text: str) -> list[RegexMatch]:
    return [RegexMatch(label, m.start(), m.end(), m.group(0))
            for m in pattern.finditer(text)]


# ── NIF ───────────────────────────────────────────────────────────────────────

_NIF_PAT = re.compile(
    r"(?<![0-9A-Z])"
    r"(?:PT\s*)?"
    r"([1-9]\d{2}[\s.]?\d{3}[\s.]?\d{3})"
    r"(?![0-9])",
    re.IGNORECASE,
)

_NIF_CONTEXT = [
    "nif", "fiscal", "contribuinte",
    "identificação", "identificacao",
]

def find_nif(text: str, require_context: bool = True) -> list[RegexMatch]:
    """ 
    accepted formats: (PT) 226154947, 226 154 947, 226.154.947 
    """
    results: list[RegexMatch] = []
    for m in _NIF_PAT.finditer(text):
        raw = re.sub(r"[\s.]", "", m.group(0).upper().removeprefix("PT").strip())
        if not _validate_nif(raw):
            continue
        if require_context and not _has_context(text, m.start(), m.end(), _NIF_CONTEXT):
            continue
        results.append(RegexMatch("NIF", m.start(), m.end(), m.group(0)))
    return results


# ── TELEMOVEL ─────────────────────────────────────────────────────────────────

_SEP   = r"[\s\-]?"
_CODE  = r"(?:\+351|351|0351)" + _SEP     

_MOB_PAT = re.compile(
    rf"(?<!\d)(?:{_CODE})?9[1236](?:{_SEP}\d){{7}}(?!\d)",
    re.IGNORECASE,
)
_LAND_PAT = re.compile(
    rf"(?<!\d)(?:{_CODE})?2\d(?:{_SEP}\d){{7}}(?!\d)",
    re.IGNORECASE,
)
_SVC_PAT = re.compile(
    rf"(?<!\d)(?:{_CODE})?[78]\d(?:{_SEP}\d){{7}}(?!\d)",
    re.IGNORECASE,
)

_SHORT_PAT = re.compile(r"(?<!\d)1(?!12\b)\d{3}(?!\d)")

_TEL_CONTEXT = [
    "contacto", "contato", "contactar", "telemovel", "telemóvel", "fax",
    "ligar", "chamada", "telefone", "telefonar"
]

def find_telemovel(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formast:
        - 9 digits: 9[1236]XXXXXXX, 2XXXXXXXX, 7/8XXXXXXXX
        - 4 digits starting with 1
        - optional before +351, 351, 0351
        - digits together, space-separated, hyphen-separated
    """
    seen: set[tuple[int, int]] = set()
    results: list[RegexMatch] = []
    for pat in (_MOB_PAT, _LAND_PAT, _SVC_PAT, _SHORT_PAT):
        for m in pat.finditer(text):
            if require_context and not _has_context(text, m.start(), m.end(), _TEL_CONTEXT):
                continue
            span = (m.start(), m.end())
            if span not in seen:
                seen.add(span)
                results.append(RegexMatch("TELEMOVEL", m.start(), m.end(), m.group(0)))
    results.sort(key=lambda r: r.start)
    return results


# ── CARTAO_CIDADAO ────────────────────────────────────────────────────────────

_CC_FULL_PAT = re.compile(
    r"(?<!\w)"
    r"(\d{8})"
    r"[\s-]?"
    r"(\d)"
    r"[\s-]?"
    r"([A-Z0-9]{2})"
    r"[\s-]?"
    r"(\d)"
    r"(?!\w)",
    re.IGNORECASE,
)

_CC_SHORT_PAT = re.compile(r"(?<!\d)\d{7,8}(?!\d)")

_CC_CONTEXT = [
    "civil", "cidadão", "cidadao", "cc", "bi"
]

def find_cartao_cidadao(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats:
        - full 12-char: validated with cheecksum 
        (8 digits, check digit, 2 alphanumeric, control digit)
        - reduced 7-8 digits: only accepted with context keywords
    """
    results: list[RegexMatch] = []

    for m in _CC_FULL_PAT.finditer(text):
        if _validate_cartao_cidadao(m.group(0)):
            results.append(RegexMatch("CARTAO_CIDADAO", m.start(), m.end(), m.group(0)))

    covered = {(r.start, r.end) for r in results}

    for m in _CC_SHORT_PAT.finditer(text):
        if any(s <= m.start() and e >= m.end() for s, e in covered):
            continue
        if require_context and _has_context(text, m.start(), m.end(), _CC_CONTEXT):
            results.append(RegexMatch("CARTAO_CIDADAO", m.start(), m.end(), m.group(0)))

    results.sort(key=lambda r: r.start)
    return results


# ── CARTAO_BANCARIO ───────────────────────────────────────────────────────────

_CARD16_PAT = re.compile(
    rf"(?<!\d)([456]\d{{3}}){_SEP}(\d{{4}}){_SEP}(\d{{4}}){_SEP}(\d{{4}})(?!\d)"
)

_AMEX_PAT = re.compile(
    rf"(?<!\d)(3[47]\d{{2}}){_SEP}(\d{{6}}){_SEP}(\d{{5}})(?!\d)"
)

_CARD_CONTEXT = [
    "crédito", "credit", "débito", "debit",
    "visa", "mastercard", "cvv", "cvc", "american express", "express",
    "cartão", "cartao", "card", "mb", "ae", "express"
]

def find_cartao_bancario(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats:
        - 16 digits starting with 4, 5 or 6, reject if any digit repeats over 4x
        - 15 digits starting with 34 or 37 (American Express)
        - consecutive digits, groups of 4 separated by space or hyphen
    both formats validated with a Luhn checksum
    """
    results: list[RegexMatch] = []

    for m in _CARD16_PAT.finditer(text):
        digits = _digits_only(m.group(0))
        if re.search(r"(\d)\1{4,}", digits):
            continue
        if not _validate_luhn(digits):
            continue
        if require_context and not _has_context(text, m.start(), m.end(), _CARD_CONTEXT):
            continue
        results.append(RegexMatch("CARTAO_BANCARIO", m.start(), m.end(), m.group(0)))

    for m in _AMEX_PAT.finditer(text):
        if not _validate_luhn(m.group(0)):
            continue
        if require_context and not _has_context(text, m.start(), m.end(), _CARD_CONTEXT):
            continue
        results.append(RegexMatch("CARTAO_BANCARIO", m.start(), m.end(), m.group(0)))

    results.sort(key=lambda r: r.start)
    return results


# ── IBAN / NIB ────────────────────────────────────────────────────────────────

_IBAN_PAT = re.compile(
    r"\b[A-Z]{2}\d{2}"
    r"(?:(?:\s?[A-Z0-9]{4}){4,7}\s?[A-Z0-9]{1,4}"   
    r"|(?:\s?\d{4}){4}\s?\d"                      
    r"|\d{21})\b",
    re.IGNORECASE,
)

_NIB_PAT = re.compile(
    r"(?<!\d)"
    r"(?:\d{4} \d{4} \d{11} \d{2}"
    r"|\d{21}"
    r"|\d{4} \d{4} \d{4} \d{4} \d{4} \d"
    r")(?!\d)"
)

_BANK_CONTEXT = ["nib", "iban", "bancária", "bancaria", "conta"]

def find_iban_nib(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats:
        - XXXX XXXX XXXX XXXX XXXX X, XXXXXXXXXXXXXXXXXXXXX, XXXX XXXX XXXXXXXXXXX XX
        - for IBAN PT50 in front
        - both validated with checksum
    """
    results: list[RegexMatch] = []

    for m in _IBAN_PAT.finditer(text):
        if _validate_iban(m.group(0)):
            results.append(RegexMatch("IBAN", m.start(), m.end(), m.group(0)))

    iban_spans = {(r.start, r.end) for r in results}

    for m in _NIB_PAT.finditer(text):
        if any(s <= m.start() and e >= m.end() for s, e in iban_spans):
            continue
        if not _validate_nib(m.group(0)):
            continue
        if require_context and not _has_context(text, m.start(), m.end(), _BANK_CONTEXT):
            continue
        results.append(RegexMatch("NIB", m.start(), m.end(), m.group(0)))

    results.sort(key=lambda r: r.start)
    return results


# ── NISS ──────────────────────────────────────────────────────────────────────

_NISS_PAT = re.compile(r"(?<!\d)\d(?:[\s.]?\d){10}(?!\d)")

_NISS_CONTEXT = ["segurança", "seguranca", "social", "niss", "seg."]

def find_niss(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats: 11 digits, with an optional single space or dot
    between any two digits (covers common groupings like 1-4-4-2,
    3-4-4, 2-3-3-3, ...) as well as no separators at all
    """
    results: list[RegexMatch] = []
    for m in _NISS_PAT.finditer(text):
        if not _validate_niss(m.group(0)):
            continue
        if require_context and not _has_context(text, m.start(), m.end(), _NISS_CONTEXT):
            continue
        results.append(RegexMatch("NISS", m.start(), m.end(), m.group(0)))
    return results


# ── PASSAPORTE ────────────────────────────────────────────────────────────────

_PASS_PAT = re.compile(r"(?<![A-Z0-9])([A-Z]{1,2})[\s\-]?(\d{6})(?!\d)", re.IGNORECASE)

_PASS_CONTEXT = ["passaport", "passport", "passaporte"]

def find_passaporte(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats: any 1/2 letters + (separator) + 6 digits
    """
    results: list[RegexMatch] = []
    for m in _PASS_PAT.finditer(text):
        if require_context and not _has_context(text, m.start(), m.end(), _PASS_CONTEXT):
            continue
        results.append(RegexMatch("PASSAPORTE", m.start(), m.end(), m.group(0)))
    return results


# ── CARTA DE CONDUÇÃO ─────────────────────────────────────────────────────────

_CART_PAT = re.compile(r"(?<![A-Z0-9])([A-Z]{1,2})[\s\-]?(\d{6,7})(?!\d)", re.IGNORECASE)

_CART_CONTEXT = [
    "licença", "licenca", "condução", "conducao", "cédula", "cedula", "carta"
]

def find_carta_conducao(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats: any 1/2 letters + (separator) + 6/7 digits
                      L2358629, L 2358629, L-2358629, AB123456
    """
    results: list[RegexMatch] = []
    for m in _CART_PAT.finditer(text):
        letters = len(re.sub(r"[^A-Za-z]", "", m.group(0)))
        digits  = len(re.sub(r"\D", "", m.group(0)))
        if not ((letters == 1 and digits == 7) or (letters == 2 and digits == 6)):
            continue
        if require_context and not _has_context(text, m.start(), m.end(), _CART_CONTEXT):
            continue
        results.append(RegexMatch("CARTA_CONDUCAO", m.start(), m.end(), m.group(0)))
    return results


# ── NUM_UTENTE (SNS) ──────────────────────────────────────────────────────────

_UTENTE_PAT = re.compile(r"(?<!\d)[34]\d{8}(?!\d)")

_UTENTE_CONTEXT = ["utente", "saúde", "saude", "sns"]

def find_num_utente(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats: 9 digits starting with 3/4
    """
    results: list[RegexMatch] = []
    for m in _UTENTE_PAT.finditer(text):
        if require_context and not _has_context(text, m.start(), m.end(), _UTENTE_CONTEXT):
            continue
        results.append(RegexMatch("NUM_UTENTE", m.start(), m.end(), m.group(0)))
    return results


# ── EMAIL ─────────────────────────────────────────────────────────────────────

_EMAIL_PAT = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

def find_email(text: str) -> list[RegexMatch]:
    """ 
    accepted formats: [xxxxxx]@[xxxx].[xxx]
    """
    return _simple("EMAIL", _EMAIL_PAT, text)


# ── CODIGO_POSTAL ─────────────────────────────────────────────────────────────

_CP_PAT = re.compile(r"\b\d{4}-\d{3}\b")

def find_codigo_postal(text: str) -> list[RegexMatch]:
    """
    accepted formats: XXXX-XXX 
    """
    return _simple("CODIGO_POSTAL", _CP_PAT, text)


# ── MATRICULA ─────────────────────────────────────────────────────────────────

# TODO talvez contexto?

_MAT_PAT = re.compile(
    r"\b(?:"
    r"[A-Z]{2}([- ]?)\d{2}\1\d{2}"     
    r"|\d{2}([- ]?)\d{2}\2[A-Z]{2}"   
    r"|\d{2}([- ]?)[A-Z]{2}\3\d{2}"   
    r"|[A-Z]{2}([- ]?)\d{2}\4[A-Z]{2}"
    r")\b",
    re.IGNORECASE,
)

_MAT_CONTEXT = ["carro", "matricula", "matrícula", "viatura", "veículo", 
                "veiculo", "automóvel", "automovel"]

def find_matricula(text: str, require_context: bool = True) -> list[RegexMatch]:
    """
    accepted formats: LL-NN-NN, NN-NN-LL, NN-LL-NN, LL-NN-LL
    separators consistent across the whole plate (all dashes, spaces, or none)
 
    """
    results: list[RegexMatch] = []
    for m in _MAT_PAT.finditer(text):
        if require_context and not _has_context(text, m.start(), m.end(), _MAT_CONTEXT):
            continue
        results.append(RegexMatch("MATRICULA", m.start(), m.end(), m.group(0)))
    return results


# ── NUM_PORTA ─────────────────────────────────────────────────────────────────

_NUM_PORTA_EXPLICIT_PAT = re.compile(
    r"\bn\.?[oº°]\s*\d+[A-Za-z]?"
    r"|\bno?\.\s*\d+[A-Za-z]?",
    re.IGNORECASE,
)

_NUM_PORTA_IMPLICIT_PAT = re.compile(
    r"(?<!\d)\b\d{1,4}[A-Za-z]?\b(?![-\d])"
)

_NUM_PORTA_CONTEXT = [
    "rua", "avenida", "av.", "travessa", "largo", "praça", "alameda",
    "estrada", "beco", "calçada", "caminho", "via", "morada",
    "endereço", "endereco"
]

def find_num_porta(
    text: str, exclude: list[tuple[int, int]] | None = None
) -> list[RegexMatch]:
    """
    accepted formats:
        - explicit: n.º 14, nº14, n° 14, no. 14
        - implicit: bare number e.g. Avenida 200

    `exclude` holds spans already claimed by other (more specific) finders —
    e.g. "CC n.º 82512748-4-DO-2" or "NIF n.º 226154947" — so the generic
    "n.º <digits>" marker doesn't swallow a citizen-card/NIF/phone/etc.
    number just because it also happens to be introduced by "n.º".
    """
    exclude = exclude or []

    def _overlaps_excluded(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in exclude)

    results: list[RegexMatch] = []

    for m in _NUM_PORTA_EXPLICIT_PAT.finditer(text):
        if _overlaps_excluded(m.start(), m.end()):
            continue
        results.append(RegexMatch("NUM_PORTA", m.start(), m.end(), m.group(0)))

    covered = {(r.start, r.end) for r in results}

    for m in _NUM_PORTA_IMPLICIT_PAT.finditer(text):
        if any(s <= m.start() and e >= m.end() for s, e in covered):
            continue
        if _overlaps_excluded(m.start(), m.end()):
            continue
        if _has_context(text, m.start(), m.end(), _NUM_PORTA_CONTEXT):
            results.append(RegexMatch("NUM_PORTA", m.start(), m.end(), m.group(0)))

    results.sort(key=lambda r: r.start)
    return results


# ── IDADE ─────────────────────────────────────────────────────────────────────

_IDADE_PAT = re.compile(
    r"(?:"
    r"\b(?P<n1>\d{1,3})\s+anos(?:\s+de\s+idade)?"   # 30 anos / 30 anos de idade
    r"|\b(?P<n2>\d{1,3})\s+de\s+idade"              # 30 de idade
    r"|idade\s+de\s+(?P<n3>\d{1,3})(?:\s+anos)?\b"  # idade de 30 / idade de 30 anos
    r"|idade\s*:?\s*(?P<n4>\d{1,3})\b"              # idade: 30
    r"|anos\s*:?\s*(?P<n5>\d{1,3})\b"
    r")",
    re.IGNORECASE,
)

_IDADE_GROUPS = ("n1", "n2", "n3", "n4", "n5")

def find_idade(text: str) -> list[RegexMatch]:
    """
    accepted formats (the surrounding keyword is required for context, but only
    the number itself is captured as the span so anonymization leaves "anos"/
    "idade" untouched):
        - 30 anos, 30 anos de idade          (number before keyword)
        - 30 de idade                        (number + de idade)
        - idade de 30, idade de 30 anos      (keyword before number)
        - idade(:) 30                          (field/form style)
        - anos(:) 30
    no separate context check needed — the age keywords are part of the match itself
    """
    results: list[RegexMatch] = []
    for m in _IDADE_PAT.finditer(text):
        for name in _IDADE_GROUPS:
            if m.group(name) is not None:
                results.append(RegexMatch("IDADE", m.start(name), m.end(name), m.group(name)))
                break
    return results


# ── DATETIME ──────────────────────────────────────────────────────────────────

_DT_PAT = re.compile(
    r"\b(?:"
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
    r"|\d{4}[/\-]\d{2}[/\-]\d{2}"
    r"|\d{1,2}\s+(?:de\s+)?\w+\s+(?:de\s+)?\d{4}"
    r"|\d{1,2}\s+de\s+\w+"
    r"|\w+\s+de\s+\d{4}"
    r"|(?:(?<=\bem\s)|(?<=\bdesde\s)|(?<=\bano\s)|(?<=\banos\s))(?:18|19|20)\d{2}"
    r"|\d{1,2}[h:\.]\d{2}m\d{2}s"
    r"|\d{1,2}[h:\.]\d{2}m"
    r"|\d{1,2}[h:\.]\d{2}h?"
    r"|\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?"
    r"|\d{1,2}h(?!\w)"
    r")\b",
    re.IGNORECASE,
)

def find_datetime(text: str) -> list[RegexMatch]:
    """
    accepted formats:   DD/MM/YY, DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
                        YYYY-MM-DD, YYYY/MM/DD
                        DD de Month de YYYY, DD de Month, Month de YYYY
                        em/desde/ano(s) YYYY (bare year right next to cue word)
                        HH:MM, HH:MM:SS, HH:MM AM/PM
                        12h14, 12h14m, 12h14m30s
                        12:14h, 12.14h, 12h
    """
    return _simple("DATETIME", _DT_PAT, text)


# ── QUANTITY ──────────────────────────────────────────────────────────────────

_QTY_PAT = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"(?:€|£|\$)\s*(?:\d{1,3}(?:[\s.,]\d{3})+|\d+)(?:[.,]\d+)?"
    r"|"
    r"(?:\d{1,3}(?:[\s.,]\d{3})+|\d+)(?:[.,]\d+)?\s*"
    r"(?:"
    r"€|£|\$"
    r"|euros?|dólares?|dollars?|libras?|pounds?"
    r"|kg|quilogramas?|g(?=\s|$|\W)|gramas?"
    r"|km|quilómetros?|m(?=\s|$|\W)|metros?|cm|mm"
    r"|m²|metros?\s+quadrados?"
    r"|l(?=\s|$|\W)|litros?|ml"
    r"|%|por\s+cento"
    r")"
    r")"
    r"(?!\w)",
    re.IGNORECASE,
)

def find_quantity(text: str) -> list[RegexMatch]:
    """ 
    accepted formats: number + (spaces) + unit 
                      numbers can be gruped by thousands with spaces, ".", ","
                      and can have decimals
                      units can be money symbols, written currency, distance, area ...
                      currency can be unit + number
    """
    return _simple("QUANTITY", _QTY_PAT, text)


# ── Pipeline ──────────────────────────────────────────────────────────────────

_FINDERS: list[Callable[[str], list[RegexMatch]]] = [
    find_email,
    find_codigo_postal,
    find_iban_nib,
    find_cartao_bancario,
    find_nif,
    find_niss,
    find_cartao_cidadao,
    find_telemovel,
    find_num_utente,
    find_passaporte,
    find_carta_conducao,
    find_matricula,
    find_num_porta,
    find_idade,
    find_datetime,
    find_quantity,
]


def find_all(text: str) -> list[RegexMatch]:
    """
    goes through every finder ordered from most-specific / longest,
    EMAIL and IBAN are first to prevent their digits being found by numberic finders
    returns a non-overlapping (longest is kept), sorted list.

    find_num_porta additionally receives the spans already claimed by every
    finder ahead of it in _FINDERS as `exclude`, since "n.º <digits>" is a
    generic marker that also introduces NIF/CC/telemóvel/etc. numbers, not
    just door numbers.
    """
    raw: list[RegexMatch] = []
    prior_spans: list[tuple[int, int]] = []
    for finder in _FINDERS:
        if finder is find_num_porta:
            matches = finder(text, exclude=prior_spans)
        else:
            matches = finder(text)
            prior_spans.extend((m.start, m.end) for m in matches)
        raw.extend(matches)

    # sort start position, then by length descending
    raw.sort(key=lambda m: (m.start, -(m.end - m.start)))

    kept: list[RegexMatch] = []
    last_end = -1
    for m in raw:
        if m.start >= last_end:
            kept.append(m)
            last_end = m.end

    return kept


# ── Human-readable replacement ────────────────────────────────────────────────

REGEX_READABLE: dict[str, str] = {
    "TELEMOVEL":       "número de telemóvel anónimo",
    "NIF":             "número de contribuinte anónimo",
    "CARTAO_BANCARIO": "número de cartão bancário anónimo",
    "CARTAO_CIDADAO":  "número de cartão de cidadão anónimo",
    "EMAIL":           "endereço de email anónimo",
    "MATRICULA":       "matrícula anónima",
    "CODIGO_POSTAL":   "código postal anónimo",
    "IBAN":            "número de IBAN anónimo",
    "NIB":             "número de NIB anónimo",
    "NISS":            "número de segurança social anónimo",
    "PASSAPORTE":      "número de passaporte anónimo",
    "CARTA_CONDUCAO":  "número de carta de condução anónima",
    "NUM_UTENTE":      "número de utente de saúde anónimo",
    "IDADE":           "idade anónima",
    "NUM_PORTA":       "número de porta anónimo",
}

REGEX_GENDER: dict[str, str] = {
    "número de telemóvel anónimo":           "m",
    "número de contribuinte anónimo":        "m",
    "número de cartão bancário anónimo":     "m",
    "número de cartão de cidadão anónimo":   "m",
    "endereço de email anónimo":             "m",
    "matrícula anónima":                     "f",
    "código postal anónimo":                 "m",
    "número de IBAN anónimo":                "m",
    "número de NIB anónimo":                 "m",
    "número de segurança social anónimo":    "m",
    "número de passaporte anónimo":          "m",
    "número de carta de condução anónima":   "m",
    "número de utente de saúde anónimo":     "m",
    "idade anónima":                         "f",
    "número de porta anónimo":               "m", 
}