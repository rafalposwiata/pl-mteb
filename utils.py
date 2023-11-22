from pathlib import Path
from typing import List
import spacy
import os
import dataclasses
from spacy_download import load_spacy


class Lemmatizer:

    def __init__(self, model_name='pl_core_news_sm'):
        self.nlp = self._load_model(model_name)

    def get_lemmas(self, texts: List[str]) -> List[List[str]]:
        docs = self.nlp.pipe(texts)
        return [[token.lemma_ for token in doc if not token.is_punct] for doc in docs]

    @staticmethod
    def _load_model(model_name):
        path = Path(f'resources/spacy/{model_name}')
        if not path.exists():
            os.makedirs(path, exist_ok=True)
            model = load_spacy(model_name)
            model.to_disk(path)
        return spacy.util.load_model_from_path(path, disable=['ner', 'parser'])


def split(samples, n):
    for i in range(0, len(samples), n):
        yield samples[i:i + n]


def from_dict(clazz, data):
    try:
        field_types = {f.name:f.type for f in dataclasses.fields(clazz)}
        return clazz(**{f:from_dict(field_types[f], data[f]) for f in data})
    except:
        return data
