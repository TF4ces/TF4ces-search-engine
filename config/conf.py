

from pathlib import Path


__WORKSPACE__ = Path(__file__).resolve().parent.parent
__ALL_MODELS__ = [
    'tfidf',
    'bm25',
    'all-mpnet-base-v2',
    'all-roberta-large-v1',
    'Intel/ColBERT-NQ',   # this has to be trained.
    'Daveee/gpl_colbert'
]

__SENTENCE_TRANSFORMERS_MODELS__ = __ALL_MODELS__[2:6]
