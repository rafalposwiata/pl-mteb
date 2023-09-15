from statistics import mean
from mteb import MTEB
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


if __name__ == '__main__':
    info_about_datasets_for_clustering()
