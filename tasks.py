from dataclasses import dataclass
from typing import List
from mteb import AbsTaskClusteringFast, AbsTaskPairClassification, AbsTaskClassification, \
    AbsTaskRetrieval
from tasks_metadata import tasks_metadata


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
    TaskInfo('SICK-R-PL', 'STS'),
    TaskInfo('CDSC-R', 'STS'),
    TaskInfo('STS22', 'STS', multilingual=True),
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
    TaskInfo('SciDefRetrieval', 'Retrieval')
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
    metadata = tasks_metadata['PPC']


class PlscClusteringS2S(AbsTaskClusteringFast):
    metadata = tasks_metadata['PlscClusteringS2S']


class PlscClusteringP2P(AbsTaskClusteringFast):
    metadata = tasks_metadata['PlscClusteringP2P']


class WikinewsPlClusteringS2S(AbsTaskClusteringFast):
    metadata = tasks_metadata['WikinewsPlClusteringS2S']


class WikinewsPlClusteringP2P(AbsTaskClusteringFast):
    metadata = tasks_metadata['WikinewsPlClusteringP2P']


class SciFieldClassification(AbsTaskClassification):
    metadata = tasks_metadata['SciField']


class SciDefRetrieval(AbsTaskRetrieval):
    metadata = tasks_metadata['SciDefRetrieval']


new_tasks = {
    'PPC': PpcPC(),
    'PlscClusteringS2S': PlscClusteringS2S(),
    'PlscClusteringP2P': PlscClusteringP2P(),
    'WikinewsPlClusteringS2S': WikinewsPlClusteringS2S(),
    'WikinewsPlClusteringP2P': WikinewsPlClusteringP2P(),
    'SciField': SciFieldClassification(),
    'SciDefRetrieval': SciDefRetrieval()
}
