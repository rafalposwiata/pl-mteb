from dataclasses import dataclass
from typing import List
from mteb import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


@dataclass
class TaskInfo:
    name: str
    task_type: str
    multilingual: bool = False


tasks: List[TaskInfo] = [
    TaskInfo('CBD', 'Classification'),
    TaskInfo('PolEmo2.0-IN', 'Classification'),
    TaskInfo('PolEmo2.0-OUT', 'Classification'),
    TaskInfo('AllegroReviews', 'Classification'),
    TaskInfo('PAC', 'Classification'),
    TaskInfo('MassiveIntentClassification', 'Classification', multilingual=True),
    TaskInfo('MassiveScenarioClassification', 'Classification', multilingual=True),
    TaskInfo('EightTagsClustering', 'Clustering'),
    TaskInfo('PlscClusteringS2S', 'Clustering'),
    TaskInfo('PlscClusteringP2P', 'Clustering'),
    TaskInfo('SICK-E-PL', 'PairClassification'),
    TaskInfo('PPC', 'PairClassification'),
    TaskInfo('CDSC-E', 'PairClassification'),
    TaskInfo('PSC', 'PairClassification'),
    TaskInfo('ArguAna-PL', 'Retrieval'),
    TaskInfo('DBPedia-PL', 'Retrieval'),
    TaskInfo('FiQA-PL', 'Retrieval'),
    TaskInfo('HotpotQA-PL', 'Retrieval'),
    TaskInfo('MSMARCO-PL', 'Retrieval'),
    TaskInfo('NFCorpus-PL', 'Retrieval'),
    TaskInfo('NQ-PL', 'Retrieval'),
    TaskInfo('Quora-PL', 'Retrieval'),
    TaskInfo('SCIDOCS-PL', 'Retrieval'),
    TaskInfo('SciFact-PL', 'Retrieval'),
    TaskInfo('TRECCOVID-PL', 'Retrieval'),
    TaskInfo('SICK-R-PL', 'STS'),
    TaskInfo('CDSC-R', 'STS'),
    TaskInfo('STS22', 'STS', multilingual=True),
    TaskInfo('STSBenchmarkMultilingualSTS', 'STS', multilingual=True)
]

tasks_and_types = {task.name: task.task_type for task in tasks}

tasks_names = list(tasks_and_types.keys())

tasks_types = ['Classification', 'Clustering', 'PairClassification', 'Retrieval', 'STS']

multilingual_tasks = [task.name for task in tasks if task.multilingual]

tasks_types_main_metric = {
    'Classification': 'accuracy',
    'Clustering': 'v_measure',
    'PairClassification': 'cos_sim.ap',
    'STS': 'cos_sim.spearman',
    'Retrieval': 'ndcg_at_10'
}


def get_main_metric(task_name) -> str:
    return tasks_types_main_metric.get(tasks_and_types.get(task_name))


def is_multilingual(task_name) -> bool:
    return task_name in multilingual_tasks


def tasks_of_type(task_type) -> List[str]:
    return [task.name for task in tasks if task.task_type == task_type]


# ---------------------- New Tasks (not yet in MTEB) ---------------------- #

class PlscClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="PlscClusteringS2S",
        description="Clustering of Polish article titles from Library of Science (https://bibliotekanauki.pl/), either "
                    "on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "PL-MTEB/plsc-clustering-s2s",
            "revision": "39bcadbac6b1eddad7c1a0a176119ce58060289a",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="cc0-1.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 17534},
        avg_character_length={"test": 84.34},
    )


class PlscClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="PlscClusteringP2P",
        description="Clustering of Polish article titles+abstracts from Library of Science "
                    "(https://bibliotekanauki.pl/), either on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "PL-MTEB/plsc-clustering-p2p",
            "revision": "8436dd4c05222778013d6642ee2f3fa1722bca9b",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Topic classification"],
        license="cc0-1.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 17537},
        avg_character_length={"test": 1023.21},
    )

new_tasks = {
    'PlscClusteringS2S': PlscClusteringS2S(),
    'PlscClusteringP2P': PlscClusteringP2P()
}
