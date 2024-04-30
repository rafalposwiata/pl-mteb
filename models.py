import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertModel

from utils import Lemmatizer

model_types = [
    'ST',   # Sentence-Transformer
    'T',    # Transformer
    'SWE'   # Static Word Embedding
]


@dataclass
class ModelInfo:
    model_name: str
    model_abbr: str = None
    model_type: str = 'ST'
    prefix: str = ''
    query_prefix: str = ''
    passage_prefix: str = ''
    multilingual: bool = False
    fp16: bool = True
    path: str = ''
    additional: Dict[str, any] = None

    def get_simple_name(self) -> str:
        return self.model_name.split('/')[-1]

    def get_abbreviation(self) -> str:
        return self.model_abbr if self.model_abbr is not None else self.get_simple_name()

    def get_additional_value(self, name, default_value=None):
        return self.additional.get(name, default_value) if self.additional is not None else default_value


class KeyedVectorsModel:

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.embedding: KeyedVectors = self._load_model(model_info)
        self._size: int = self.embedding.vector_size
        self.pooling = model_info.get_additional_value('pooling')
        self.pooling_op = {'avg': self.avg_pool, 'max': self.max_pool, 'concat': self.concat_pool}[self.pooling]
        self.lemmatizer = Lemmatizer()

    @staticmethod
    def _load_model(model_info: ModelInfo) -> KeyedVectors:
        text_format: bool = model_info.path.endswith(".txt")
        model = KeyedVectors.load_word2vec_format(model_info.path, binary=False) if text_format \
            else KeyedVectors.load(model_info.path)
        if isinstance(model, Word2Vec):
            return model.wv
        return model

    def encode(self, sentences, batch_size=32, **kwargs):
        vectors = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            vectors += [self._encode(lemmas) for lemmas in self.lemmatizer.get_lemmas(batch)]

        vectors = np.stack(vectors, axis=0)
        if kwargs.get('convert_to_tensor', False):
            vectors = torch.from_numpy(vectors.astype(np.float32))
        return vectors

    def _encode(self, words: List[str]):
        sentvec = [self._vocab_vector(word.lower()) for word in words]
        sentvec = [vec for vec in sentvec if vec is not None]
        if not sentvec:
            sentvec.append(np.zeros(self._size))
        return self.pooling_op(sentvec)

    def _vocab_vector(self, word: str):
        if word in self.embedding:
            vec = self.embedding[word]
            return self.normalize(vec)
        else:
            return None

    @staticmethod
    def avg_pool(sentvec):
        return np.mean(sentvec, 0)

    @staticmethod
    def max_pool(sentvec):
        return np.max(sentvec, 0)

    def concat_pool(self, sentvec):
        return np.hstack((self.avg_pool(sentvec), self.max_pool(sentvec)))

    @staticmethod
    def normalize(vec):
        return vec / np.linalg.norm(vec)


class TransformerModel:

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        torch_type = torch.float16 if model_info.fp16 else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_info.model_name, torch_dtype=torch_type)
        self.model = AutoModel.from_pretrained(model_info.model_name)

    def encode(self, sentences, batch_size=32, **kwargs):
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            embeddings += self._encode(batch)

        if kwargs.get('convert_to_tensor', False):
            embeddings = torch.stack(embeddings)
        else:
            embeddings = np.asarray([emb.numpy() for emb in embeddings])
        return embeddings

    def _encode(self, batch):
        max_length = self.model_info.get_additional_value('max_length', 512)
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        with torch.no_grad():
            return self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output


class ModelWrapper:

    def __init__(self, model, model_info: ModelInfo):
        self.model = model
        self.model_info = model_info

    def encode(self, sentences, batch_size=32, **kwargs):
        sentences = ['{}{}'.format(self.model_info.prefix, sentence) for sentence in sentences]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)


class RetrievalModelWrapper(DRESModel):

    def __init__(self, model, model_info: ModelInfo, **kwargs):
        super().__init__(model, **kwargs)
        self.model_info = model_info

    def encode_queries(self, queries: List[Union[str, Dict]], batch_size: int, **kwargs):
        queries = ['{}{}'.format(self.model_info.query_prefix, q if isinstance(q, str) else q.get('text', ''))
                   for q in queries]
        return self.model.encode(queries, batch_size=batch_size, normalize_embeddings=True, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        passages = ['{}{} {}'.format(self.model_info.passage_prefix, doc.get('title', ''),
                                     doc['text']).strip() for doc in corpus]
        return self.model.encode(passages, batch_size=batch_size, normalize_embeddings=True, **kwargs)
