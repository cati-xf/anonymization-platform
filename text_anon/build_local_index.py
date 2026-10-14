"""
One-time setup script: downloads all Portuguese-related Wikidata entities and
stores them in anon_kb.db (local_entity_index + local_substitution_pool).

Run once:  python -m text_anon.build_local_index

Coverage — three broad Wikidata properties × three type properties:
  Entity scope:  P17=Q45 (in Portugal) | P27=Q45 (citizenship) | P495=Q45 (origin)
  Type lookup:   P31 (instance of)     | P279 (subclass of)    | P361 (part of)

Each combination is a separate required-join query (no OPTIONAL UNION) so
Wikidata can use its indexes efficiently and avoid gateway timeouts.

These queries are categorical — they never expose any individual being
anonymized, so running them at setup is safe from a privacy standpoint.
"""

import re
import sqlite3
import time

import requests

from .replacement.db import _DB_PATH, _get_conn
from .replacement.wikidata import _gender_from_label

_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
_HEADERS = {
    "User-Agent": "TextAnonymizer/LocalBuild (one-time setup; not live anonymization)",
    "Accept": "application/sparql-results+json",
}
_PAGE_SIZE = 5_000
_DELAY     = 2.0   # seconds between pages
_RETRIES   = 3
_QID_RE    = re.compile(r"^Q\d+$")


def _gender_from_uri(uri: str) -> str | None:
    qid = uri.rsplit("/", 1)[-1]
    if qid == "Q499327":
        return "m"
    if qid == "Q1775415":
        return "f"
    return None


def _sparql(query: str, timeout: int = 90) -> list[dict]:
    for attempt in range(_RETRIES):
        try:
            r = requests.get(
                _SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=_HEADERS,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:
            if attempt < _RETRIES - 1:
                wait = 15 * (2 ** attempt)   # 15 s, 30 s, …
                print(f"  error (attempt {attempt + 1}/{_RETRIES}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  SPARQL failed after {_RETRIES} attempts: {e}")
                return []
    return []


def _store_row(conn: sqlite3.Connection, row: dict) -> bool:
    label = row.get("itemLabel", {}).get("value", "")
    if not label or _QID_RE.match(label):
        return False

    qid_uri = row.get("item", {}).get("value", "")
    qid = qid_uri.rsplit("/", 1)[-1] if qid_uri else ""
    if not qid:
        return False

    type_uri   = row.get("type", {}).get("value", "")
    type_qid   = type_uri.rsplit("/", 1)[-1] if type_uri else None
    type_label = row.get("typeLabel", {}).get("value") or None
    if type_label and _QID_RE.match(type_label):
        type_label = None

    gender_uri = row.get("gender", {}).get("value", "")
    gender = _gender_from_uri(gender_uri) if gender_uri else _gender_from_label(label)

    conn.execute(
        "INSERT OR REPLACE INTO local_entity_index "
        "(label_norm, qid, type_qid, type_label, gender) VALUES (?,?,?,?,?)",
        (label.lower().strip(), qid, type_qid, type_label, gender),
    )
    if type_qid:
        conn.execute(
            "INSERT OR REPLACE INTO local_substitution_pool (type_qid, label, gender) VALUES (?,?,?)",
            (type_qid, label, gender),
        )
    return True


def _make_query(entity_prop: str, type_prop: str, last_qid: int,
                extra_gender_prop: str | None = None) -> str:
    extra = f"  OPTIONAL {{ ?item wdt:{extra_gender_prop} ?gender . }}\n" if extra_gender_prop else ""
    cursor = f"  FILTER(?item > wd:Q{last_qid})\n" if last_qid > 0 else ""
    return f"""
SELECT ?item ?itemLabel ?type ?typeLabel ?gender WHERE {{
  ?item wdt:{entity_prop} wd:Q45 .
  ?item wdt:{type_prop} ?type .
{cursor}  hint:Prior hint:rangeSafe true .
  OPTIONAL {{ ?item wdt:P5185 ?gender . }}
{extra}  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en". }}
}}
ORDER BY ?item
LIMIT {_PAGE_SIZE}
"""


def _last_qid_from_page(rows: list[dict]) -> int:
    last_uri = rows[-1].get("item", {}).get("value", "")
    try:
        return int(last_uri.rsplit("Q", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _fetch_all(conn: sqlite3.Connection, description: str,
               entity_prop: str, type_prop: str,
               extra_gender_prop: str | None = None) -> int:
    total = 0
    last_qid = 0
    while True:
        rows = _sparql(_make_query(entity_prop, type_prop, last_qid, extra_gender_prop))
        if not rows:
            break
        stored = sum(_store_row(conn, r) for r in rows)
        total += stored
        conn.commit()
        if len(rows) < _PAGE_SIZE:
            break
        last_qid = _last_qid_from_page(rows)
        print(f"    {description}: {total:,} so far (cursor Q{last_qid})…")
        time.sleep(_DELAY)
    return total


# (description, entity_prop, type_prop, extra_gender_prop)
# P27 (citizens) also checks P21 for biological sex as gender fallback
QUERIES = [
    ("P17=Q45 + P31  (in Portugal, instance of)",  "P17",  "P31",  None),
    ("P17=Q45 + P279 (in Portugal, subclass of)",  "P17",  "P279", None),
    ("P17=Q45 + P361 (in Portugal, part of)",      "P17",  "P361", None),
    ("P27=Q45 + P31  (citizen, instance of)",       "P27",  "P31",  "P21"),
    ("P27=Q45 + P279 (citizen, subclass of)",       "P27",  "P279", "P21"),
    ("P27=Q45 + P361 (citizen, part of)",           "P27",  "P361", "P21"),
    ("P495=Q45 + P31  (pt origin, instance of)",   "P495", "P31",  None),
    ("P495=Q45 + P279 (pt origin, subclass of)",   "P495", "P279", None),
    ("P495=Q45 + P361 (pt origin, part of)",       "P495", "P361", None),
]


def main() -> None:
    conn = _get_conn()

    print("=== Building local Wikidata snapshot ===\n")
    print(f"Database: {_DB_PATH}\n")

    for desc, entity_prop, type_prop, extra in QUERIES:
        print(f"-- {desc} --")
        n = _fetch_all(conn, desc, entity_prop, type_prop, extra)
        print(f"  → {n:,} rows stored\n")
        time.sleep(_DELAY)

    ei    = conn.execute("SELECT COUNT(*) FROM local_entity_index").fetchone()[0]
    sp    = conn.execute("SELECT COUNT(*) FROM local_substitution_pool").fetchone()[0]
    types = conn.execute("SELECT COUNT(DISTINCT type_qid) FROM local_substitution_pool").fetchone()[0]
    conn.close()

    print("=== Done ===")
    print(f"  local_entity_index:      {ei:,} entries")
    print(f"  local_substitution_pool: {sp:,} entries across {types:,} distinct types")
    print("\nSet PRIVACY_MODE=true to never fall back to live Wikidata queries.")


if __name__ == "__main__":
    main()
