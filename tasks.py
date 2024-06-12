from dataclasses import dataclass
from typing import List
from mteb import AbsTaskClusteringFast, TaskMetadata, AbsTaskPairClassification, AbsTaskClassification, \
    AbsTaskRetrieval


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
    TaskInfo('SciField', 'Classification'),
    TaskInfo('EightTagsClustering', 'Clustering'),
    TaskInfo('PlscClusteringS2S', 'Clustering'),
    TaskInfo('PlscClusteringP2P', 'Clustering'),
    TaskInfo('WikinewsPlClusteringS2S', 'Clustering'),
    TaskInfo('WikinewsPlClusteringP2P', 'Clustering'),
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
    TaskInfo('SciDefRetrieval', 'Retrieval'),
    TaskInfo('SICK-R-PL', 'STS'),
    TaskInfo('CDSC-R', 'STS'),
    TaskInfo('STS22', 'STS', multilingual=True),
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


# ---------------------- Because of compatibility issues with newer versions of MTEB  ---------------------- #


class PpcPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PPC",
        dataset={
            "path": "PL-MTEB/ppc-pairclassification",
            "revision": "2c7d2df57801a591f6b1e3aaf042e7a04ec7d9f2",
        },
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2207.12759.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )


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


class WikinewsPlClusteringS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="WikinewsPlClusteringS2S",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/wikinews-pl",
        dataset={
            "path": "PL-MTEB/wikinews-pl-clustering-s2s",
            "revision": "2355f412aa5b735fc66cbfe8dff766c49fb45a5d",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-06-05"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-2.5",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 4477},
        avg_character_length={"test": 60.18},
    )


class WikinewsPlClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="WikinewsPlClusteringP2P",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/wikinews-pl",
        dataset={
            "path": "PL-MTEB/wikinews-pl-clustering-p2p",
            "revision": "8f10d8e9d094ff9f2a8e99e645ad22eb745eb3cf",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-06-05"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-2.5",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 4477},
        avg_character_length={"test": 1127.31},
    )


class SciFieldClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SciField",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/open-coursebooks-pl",
        dataset={
            "path": "PL-MTEB/scifield",
            "revision": "8064b4442da038aa0482a2542370f9a2441cc50a",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="cc-by-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"train": 5751, "test": 1438},
        avg_character_length={"train": 431.6, "test": 430.4},
    )


class SciDefRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciDefRetrieval",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/open-coursebooks-pl",
        dataset={
            "path": "PL-MTEB/scidef",
            "revision": "4755de8baebc07bed0c60cec9f796e9a346600cb",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )


new_tasks = {
    'PPC': PpcPC(),
    'PlscClusteringS2S': PlscClusteringS2S(),
    'PlscClusteringP2P': PlscClusteringP2P(),
    'WikinewsPlClusteringS2S': WikinewsPlClusteringS2S(),
    'WikinewsPlClusteringP2P': WikinewsPlClusteringP2P(),
    'SciField': SciFieldClassification(),
    'SciDefRetrieval': SciDefRetrieval()
}
