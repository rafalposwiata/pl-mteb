from dataclasses import dataclass
from typing import List


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


new_tasks = {}
