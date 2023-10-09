from dataclasses import dataclass
from typing import List, Dict
from mteb import DRESModel


@dataclass
class ModelInfo:
    model_name: str
    model_abbr: str = None
    prefix: str = ''
    query_prefix: str = ''
    passage_prefix: str = ''
    multilingual: bool = False
    fp16: bool = True

    def get_simple_name(self) -> str:
        return self.model_name.split('/')[-1]

    def get_abbreviation(self) -> str:
        return self.model_abbr if self.model_abbr is not None else self.get_simple_name()


models: List[ModelInfo] = [
    ModelInfo('sdadas/st-polish-paraphrase-from-mpnet', 'st-polish-para-mpnet'),
    ModelInfo('sdadas/st-polish-paraphrase-from-distilroberta', 'st-polish-para-distilroberta'),
    ModelInfo('ipipan/herbert-base-retrieval-v2'),
    ModelInfo('distiluse-base-multilingual-cased-v2', 'distiluse-base-multi-cased-v2', multilingual=True, fp16=False),
    ModelInfo('paraphrase-multilingual-MiniLM-L12-v2', 'para-multi-MiniLM-L12-v2', multilingual=True),
    ModelInfo('paraphrase-multilingual-mpnet-base-v2', 'para-multi-mpnet-base-v2', multilingual=True),
    ModelInfo('LaBSE', multilingual=True),
    ModelInfo('intfloat/multilingual-e5-large', 'multi-e5-large', multilingual=True, prefix='query: ',
              query_prefix='query: ', passage_prefix='passage: '),
    ModelInfo('intfloat/multilingual-e5-base', 'multi-e5-base', multilingual=True, prefix='query: ',
              query_prefix='query: ', passage_prefix='passage: '),
    ModelInfo('intfloat/multilingual-e5-small', 'multi-e5-small', multilingual=True, prefix='query: ',
              query_prefix='query: ', passage_prefix='passage: ')
]

models_abbreviations = {model.get_simple_name(): model.get_abbreviation() for model in models}


class Model:

    def __init__(self, model, model_info):
        self.model = model
        self.model_info = model_info

    def encode(self, sentences, batch_size=32, **kwargs):
        sentences = ['{}{}'.format(self.model_info.prefix, sentence) for sentence in sentences]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)


class RetrievalModel(DRESModel):

    def __init__(self, model, model_info: ModelInfo, **kwargs):
        super().__init__(model, **kwargs)
        self.model_info = model_info

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        queries = ['{}{}'.format(self.model_info.query_prefix, q.get('text', '')) for q in queries]
        return self.model.encode(queries, batch_size=batch_size, normalize_embeddings=True, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        passages = ['{}{} {}'.format(self.model_info.passage_prefix, doc.get('title', ''),
                                     doc['text']).strip() for doc in corpus]
        return self.model.encode(passages, batch_size=batch_size, normalize_embeddings=True, **kwargs)
