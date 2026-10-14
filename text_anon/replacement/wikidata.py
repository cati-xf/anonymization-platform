import re
import json

import requests

from .db import _get_conn, _PRIVACY_MODE
from .random_person import _MASC_ARTICLES, _FEM_ARTICLES


def _wikidata_search(term: str, lang: str = "pt") -> str | None:
    """
    Searches wikidata for term and returns QID
    """
    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": term,
                "language": lang,
                "format": "json",
                "limit": 5,
            },
            headers={"User-Agent": "Text Anonymizer"},
            timeout=5,
        )
        results = r.json().get("search", [])
        if results:
            return results[0]["id"]
    except Exception:
        pass
    return None


def _wikidata_entity(qid: str) -> dict:
    """
    Gets entity JSON from QID
    """
    try:
        r = requests.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
            headers={"User-Agent": "TextAnonymizer"},
            timeout=5,
        )
        return r.json()["entities"].get(qid, {})
    except Exception:
        return {}


def _gender_from_qid(qid: str) -> str | None:
    if qid == "Q499327":
        return "m"
    if qid == "Q1775415":
        return "f"
    return None


def _gender_from_entity(entity: dict) -> str | None:
    """
    Checks grammatical gender and sex of a person
    """
    claims = entity.get("claims", {})
    for prop in ("P5185", "P21"):
        if prop in claims:
            try:
                val = claims[prop][0]["mainsnak"]["datavalue"]["value"]
                qid = val.get("id", "")
                g = _gender_from_qid(qid)
                if g:
                    return g
            except (KeyError, IndexError):
                pass
    return None


def _gender_from_label(label: str) -> str | None:
    """
    Fallback if wikidata does not have gender
    """
    l = label.lower().split()[0]
    if l.endswith(("a", "ade", "ção", "são", "gem", "ise", "ite", "dade")):
        return "f"
    if l.endswith(("o", "or", "ão", "ema", "ume", "il", "al", "ar")):
        return "m"
    _known = {
        "hospital": "m", "museu": "m", "banco": "m", "tribunal": "m",
        "cidade": "f", "vila": "f", "aldeia": "f", "parish": "f",
    }
    return _known.get(l)


def _pt_label(entity: dict) -> str | None:
    """
    For fallback use own label
    """
    labels = entity.get("labels", {})
    pt = labels.get("pt") or labels.get("en")
    return pt["value"] if pt else None


def _headword_pt(surface: str) -> str | None:
    """First meaningful token of a PT surface, skipping stopwords and single chars."""
    _stop = _MASC_ARTICLES | _FEM_ARTICLES | {"de", "para", "por", "com", "e"}
    for token in surface.lower().split():
        if token not in _stop and len(token) > 1:
            return token
    return None


def _get_parent_label_and_gender(term: str, ner_label: str) -> tuple[str | None, str | None]:
    """
    Looks for parent (subclass, instance, part of) and associates gender.
    Falls back to headword lookup when the full term is not found in Wikidata.
    """
    qid = _wikidata_search(term)
    if not qid:
        hw = _headword_pt(term)
        if hw and hw != term.lower():
            qid = _wikidata_search(hw)
    if not qid:
        return None, None
    entity = _wikidata_entity(qid)
    if not entity:
        return None, None

    claims = entity.get("claims", {})
    parent_qid = None
    for prop in ["P279", "P31", "P361"]:
        if prop in claims:
            try:
                parent_qid = claims[prop][0]["mainsnak"]["datavalue"]["value"]["id"]
                break
            except (KeyError, IndexError):
                pass

    if not parent_qid:
        label = _pt_label(entity)
        gender = _gender_from_entity(entity) or (label and _gender_from_label(label))
        return label, gender

    parent = _wikidata_entity(parent_qid)
    label  = _pt_label(parent)
    gender = _gender_from_entity(parent) or (label and _gender_from_label(label))
    return label, gender


def lookup_gen(term: str, ner_label: str, local_only: bool = False) -> tuple[str | None, str | None, bool]:
    """
    Returns (pt_label, gender, had_local) where had_local=True means a label was found
    without a live Wikidata call.  Pass local_only=True to suppress the live fallback.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT pt_label, gender FROM kb_cache WHERE term=? AND ner_label=?",
        (term.lower(), ner_label),
    ).fetchone()
    if row is not None:
        conn.close()
        return row[0], row[1], True

    local = conn.execute(
        "SELECT type_label, gender FROM local_entity_index WHERE label_norm=?",
        (term.lower().strip(),),
    ).fetchone()
    if local is not None:
        pt_label, gender = local[0], local[1]
        if pt_label and not gender:
            gender = _gender_from_label(pt_label)
        conn.execute(
            "INSERT OR REPLACE INTO kb_cache (term, ner_label, pt_label, gender) VALUES (?,?,?,?)",
            (term.lower(), ner_label, pt_label, gender),
        )
        conn.commit()
        conn.close()
        return pt_label, gender, True

    if _PRIVACY_MODE or local_only:
        conn.close()
        return None, None, False

    pt_label, gender = _get_parent_label_and_gender(term, ner_label)
    if pt_label and not gender:
        gender = _gender_from_label(pt_label)

    conn.execute(
        "INSERT OR REPLACE INTO kb_cache (term, ner_label, pt_label, gender) VALUES (?,?,?,?)",
        (term.lower(), ner_label, pt_label, gender),
    )
    conn.commit()
    conn.close()
    return pt_label, gender, False


def _sparql_query(query: str, timeout: int = 10) -> list[dict]:
    """
    Sends query and returns result rows
    """
    try:
        r = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers={
                "User-Agent": "Text Anonymizer",
                "Accept":     "application/sparql-results+json",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    except Exception:
        return []


def _get_type_qid(term: str) -> tuple[str | None, str | None]:
    """
    Finds own QID and type QID
    """
    entity_qid = _wikidata_search(term)
    if not entity_qid:
        return None, None
    entity = _wikidata_entity(entity_qid)
    claims = entity.get("claims", {})
    for prop in ("P31", "P279"):
        if prop in claims:
            try:
                type_qid = claims[prop][0]["mainsnak"]["datavalue"]["value"]["id"]
                return entity_qid, type_qid
            except (KeyError, IndexError):
                pass
    return entity_qid, None


def _get_country_qid(entity_qid: str) -> str | None:
    entity = _wikidata_entity(entity_qid)
    try:
        return entity["claims"]["P17"][0]["mainsnak"]["datavalue"]["value"]["id"]
    except (KeyError, IndexError):
        return None


def _extract_candidate(row: dict) -> tuple[str, str | None] | None:
    """
    Pulls (label, gender) from a SPARQL bindings row.
    Prefers P5185 from Wikidata; falls back to rule-based if missing.
    Returns None if label is a bare QID (means no PT/EN label was available).
    """
    label = row.get("itemLabel", {}).get("value", "")
    if not label or re.fullmatch(r"Q\d+", label):
        return None

    gender: str | None = None
    gender_uri = row.get("gender", {}).get("value", "")
    if gender_uri:
        gender = _gender_from_qid(gender_uri.rsplit("/", 1)[-1])
    if gender is None:
        gender = _gender_from_label(label)
    return label, gender


def _fetch_loc_substitutes(
    term: str, entity_qid: str, type_qid: str, n: int = 20
) -> list[tuple[str, str | None]]:
    """
    Builds query to find locations of the same type in the same country as the original entity.
    Falls back to any administrative territorial entity in that country.
    If no country is known, searches globally.
    """
    country_qid = _get_country_qid(entity_qid)

    if country_qid:
        query = f"""
        SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
          ?item wdt:P31 wd:{type_qid} .
          ?item wdt:P17 wd:{country_qid} .
          OPTIONAL {{ ?item wdt:P5185 ?gender . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
        }} LIMIT {n}
        """
    else:
        query = f"""
        SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
          ?item wdt:P31 wd:{type_qid} .
          OPTIONAL {{ ?item wdt:P5185 ?gender . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
        }} LIMIT {n}
        """
    results = _sparql_query(query)
    candidates = [c for r in results if (c := _extract_candidate(r))]

    if len(candidates) < 3:
        if country_qid:
            fallback = f"""
            SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
              ?item wdt:P17 wd:{country_qid} .
              ?item wdt:P31 ?type .
              ?type wdt:P279* wd:Q56061 .
              OPTIONAL {{ ?item wdt:P5185 ?gender . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
            }} LIMIT {n}
            """
        else:
            fallback = f"""
            SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
              ?item wdt:P31 ?type .
              ?type wdt:P279* wd:Q56061 .
              OPTIONAL {{ ?item wdt:P5185 ?gender . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
            }} LIMIT {n}
            """
        results = _sparql_query(fallback)
        candidates = [c for r in results if (c := _extract_candidate(r))]
    return candidates


def _fetch_org_substitutes(
    term: str, entity_qid: str, type_qid: str, n: int = 20
) -> list[tuple[str, str | None]]:
    """
    Builds query to find list of organizations, same type in Portugal first.
    If less than 3 results, looks globally.
    """
    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
      ?item wdt:P31 wd:{type_qid} .
      ?item wdt:P17 wd:Q45 .
      OPTIONAL {{ ?item wdt:P5185 ?gender . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
    }} LIMIT {n}
    """
    results = _sparql_query(query)
    candidates = [c for r in results if (c := _extract_candidate(r))]
    if len(candidates) < 3:
        query_global = f"""
        SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
          ?item wdt:P31 wd:{type_qid} .
          OPTIONAL {{ ?item wdt:P5185 ?gender . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
        }} LIMIT {n}
        """
        results = _sparql_query(query_global)
        candidates = [c for r in results if (c := _extract_candidate(r))]
    return candidates


def _fetch_misc_substitutes(
    term: str, entity_qid: str, type_qid: str, n: int = 20
) -> list[tuple[str, str | None]]:
    """
    Builds query to find list of misc entities, same type.
    If less than 3 results, broadens to any entity whose type is a subclass.
    """
    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
      ?item wdt:P31 wd:{type_qid} .
      OPTIONAL {{ ?item wdt:P5185 ?gender . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
    }} LIMIT {n}
    """
    results = _sparql_query(query)
    candidates = [c for r in results if (c := _extract_candidate(r))]
    if len(candidates) < 3:
        fallback_query = f"""
        SELECT DISTINCT ?item ?itemLabel ?gender WHERE {{
          ?item wdt:P31 ?type .
          ?type wdt:P279* wd:{type_qid} .
          OPTIONAL {{ ?item wdt:P5185 ?gender . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
        }} LIMIT {n}
        """
        results = _sparql_query(fallback_query)
        candidates = [c for r in results if (c := _extract_candidate(r))]
    return candidates


def wikidata_substitutes(
    term: str, ner_label: str, n: int = 20, local_only: bool = False
) -> tuple[list[tuple[str, str | None]], bool]:
    """
    Returns (candidates, had_local) where had_local=True means no live SPARQL
    call was made (candidates came entirely from the local pool or cache).
    If the local pool has fewer than 3 candidates, a live call is made and its
    results are merged into (not substituted for) the local ones; had_local is
    False whenever that live call is actually issued, regardless of whether it
    returns anything new.
    Pass local_only=True to suppress the live fallback.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT candidates FROM substitution_cache WHERE term=? AND ner_label=?",
        (term.lower(), ner_label),
    ).fetchone()
    if row is not None:
        conn.close()
        return [(name, gender) for name, gender in json.loads(row[0])], True

    local_idx = conn.execute(
        "SELECT qid, type_qid FROM local_entity_index WHERE label_norm=?",
        (term.lower().strip(),),
    ).fetchone()

    candidates: list[tuple[str, str | None]] = []

    if local_idx and local_idx[1]:
        type_qid = local_idx[1]
        pool_rows = conn.execute(
            "SELECT label, gender FROM local_substitution_pool WHERE type_qid=? LIMIT ?",
            (type_qid, n),
        ).fetchall()
        candidates = [(r[0], r[1]) for r in pool_rows]

    had_local = bool(candidates)

    if len(candidates) < 3 and not _PRIVACY_MODE and not local_only:
        if local_idx:
            entity_qid_live, type_qid_live = local_idx[0], local_idx[1]
        else:
            entity_qid_live, type_qid_live = _get_type_qid(term)
        if entity_qid_live and type_qid_live:
            live_candidates: list[tuple[str, str | None]] = []
            if ner_label == "LOC":
                live_candidates = _fetch_loc_substitutes(term, entity_qid_live, type_qid_live, n)
            elif ner_label == "ORG":
                live_candidates = _fetch_org_substitutes(term, entity_qid_live, type_qid_live, n)
            elif ner_label == "MISC":
                live_candidates = _fetch_misc_substitutes(term, entity_qid_live, type_qid_live, n)

            if ner_label in ("LOC", "ORG", "MISC"):
                # A live SPARQL call was issued -- merge into (not overwrite)
                # the local candidates, and report had_local=False because a
                # live round-trip is what this call actually paid for,
                # regardless of whether it happened to add anything new.
                had_local = False
                seen = {name.lower() for name, _ in candidates}
                for name, gender in live_candidates:
                    if name.lower() not in seen:
                        candidates.append((name, gender))
                        seen.add(name.lower())
                candidates = candidates[:n]

    conn.execute(
        "INSERT OR REPLACE INTO substitution_cache (term, ner_label, candidates) VALUES (?,?,?)",
        (term.lower(), ner_label,
         json.dumps([[name, gender] for name, gender in candidates], ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    return candidates, had_local
