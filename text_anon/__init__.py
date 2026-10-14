from .anon_types import AnonymizationConfig, AnonSpan, VALID_METHODS
from .replacement import SUBSTITUTION_POOL_WITH_GENDER
from .article import GENDER, ARTICLES, BARE_PREP_CONTRACTION, fix_article
from .replacement import lookup_gen, wikidata_substitutes
from .pipeline import (
    pipeline_with_spans,
    detect_entities,
    lookup_live,
    apply_entities,
)
