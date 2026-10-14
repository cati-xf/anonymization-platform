import os
import sqlite3

# package root (text_anon/), where anon_kb.db actually lives
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PACKAGE_ROOT, "anon_kb.db")

_PRIVACY_MODE: bool = os.environ.get("PRIVACY_MODE", "").lower() in ("1", "true", "yes")


def _get_conn() -> sqlite3.Connection:
    """
    Creates tables to cache wikidata generalization results and substitution candidates
    """
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_cache (
            term        TEXT NOT NULL,
            ner_label   TEXT NOT NULL,
            pt_label    TEXT,
            gender      TEXT,
            PRIMARY KEY (term, ner_label)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS substitution_cache (
            term        TEXT NOT NULL,
            ner_label   TEXT NOT NULL,
            candidates  TEXT NOT NULL,
            PRIMARY KEY (term, ner_label)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS local_entity_index (
            label_norm  TEXT PRIMARY KEY,
            qid         TEXT NOT NULL,
            type_qid    TEXT,
            type_label  TEXT,
            gender      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS local_substitution_pool (
            type_qid    TEXT NOT NULL,
            label       TEXT NOT NULL,
            gender      TEXT,
            PRIMARY KEY (type_qid, label)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pool_type ON local_substitution_pool(type_qid)")
    conn.commit()
    return conn
