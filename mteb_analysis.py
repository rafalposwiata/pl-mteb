from statistics import mean
from beir.datasets.data_loader_hf import HFDataLoader
from mteb import MTEB
from langdetect import detect
from tqdm import tqdm
import datasets


def info_about_datasets_for_clustering() -> None:
    _mteb = MTEB(task_types=['Clustering'])
    result = {}
    for task in _mteb.tasks:
        dataset = datasets.load_dataset(task.description['hf_hub_name'], ignore_verifications=True)
        sentences_per_row = [len(row['sentences']) for row in dataset['test']]
        result[task.description["name"]] = {"rows": len(dataset["test"]), "avg_sentences": int(mean(sentences_per_row))}

    for name, stats in result.items():
        print(f'{name} - {stats["rows"]} / {stats["avg_sentences"]}')


def beir_pl_analysis() -> None:
    _mteb = MTEB(task_types=['Retrieval'], task_langs=['pl'])
    result = {}

    def is_non_polish(text) -> bool:
        try:
            if text != '' and detect(text) != 'pl':
                return True
        except:
            pass
        return False

    def format_result(no_non_polish_texts, no_docs) -> str:
        return f'{no_non_polish_texts} ({round(no_non_polish_texts / no_docs, 3)})'

    for task in _mteb.tasks:
        dataset_name = task.description['beir_name']
        print(dataset_name)
        if dataset_name == 'trec-covid-pl':
            corpus, queries, qrels = HFDataLoader(hf_repo=f"clarin-knext/{dataset_name}", streaming=False,
                                                  keep_in_memory=False).load(split='test')

            non_polish_queries = [query['text'] for query in tqdm(queries) if is_non_polish(query['text'])]
            non_polish_corpus_titles = [doc['title'] for doc in tqdm(corpus) if is_non_polish(doc['title'])]
            non_polish_corpus_texts = [doc['text'] for doc in tqdm(corpus) if is_non_polish(doc['text'])]

            result[dataset_name] = {'non_polish_queries': format_result(len(non_polish_queries), len(queries)),
                                    'non_polish_corpus_titles': format_result(len(non_polish_corpus_titles),
                                                                              len(corpus)),
                                    'non_polish_corpus_texts': format_result(len(non_polish_corpus_texts), len(corpus))}

    for name, stats in result.items():
        print(f'{name} - {stats}')


if __name__ == '__main__':
    info_about_datasets_for_clustering()
    beir_pl_analysis()
