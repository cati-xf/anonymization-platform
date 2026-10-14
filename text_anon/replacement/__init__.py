from .dispatch import (
    generalize_ne,
    make_regex_replacement,
    make_ne_replacement,
    make_unique_rand_replacement,
    number_duplicate_gen_replacements,
    lookup_live,
)
from .random_person import SUBSTITUTION_POOL_WITH_GENDER, infer_person_gender, pick_random_person
from .wikidata import lookup_gen, wikidata_substitutes
