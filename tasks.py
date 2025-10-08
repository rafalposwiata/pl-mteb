from typing import List
from mteb import AbsTaskClusteringFast, AbsTaskClassification, AbsTaskRetrieval, overview
from mteb.overview import MTEBTasks, get_task
from tasks_metadata import tasks_metadata

tasks: dict[str, List[str]] = {
    "Classification": [
        "CBD.v2",
        "PolEmo2.0-IN.v2",
        "PolEmo2.0-OUT.v2",
        "AllegroReviews.v2",
        "PAC.v2",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "SciField"
    ],
    "Clustering": [
        "EightTagsClustering.v2",
        "PlscClusteringS2S.v2",
        "PlscClusteringP2P.v2",
        "WikinewsPlClusteringS2S",
        "WikinewsPlClusteringP2P"
    ],
    "PairClassification": [
        "SICK-E-PL",
        "PpcPC",
        "CDSC-E",
        "PSC"
    ],
    "STS": [
        "SICK-R-PL",
        "CDSC-R",
        "STS22.v2"
    ],
    "Retrieval": [
        "ArguAna-PL",
        "DBPedia-PLHardNegatives",
        "FiQA-PL",
        "HotpotQA-PLHardNegatives",
        "MSMARCO-PLHardNegatives",
        "NFCorpus-PL",
        "NQ-PLHardNegatives",
        "Quora-PLHardNegatives",
        "SCIDOCS-PL",
        "SciFact-PL",
        "TRECCOVID-PL",
        "SciDefRetrieval",
    ]
}

tasks_and_types = {task_name: task_type for task_type, task_names in tasks.items() for task_name in task_names}

tasks_names = list(tasks_and_types.keys())

tasks_types_main_metric = {
    "Classification": "accuracy",
    "Clustering": "v_measure",
    "PairClassification": "cos_sim.ap",
    "STS": "cos_sim.spearman",
    "Retrieval": "ndcg_at_10"
}


def get_main_metric(task_name) -> str:
    return tasks_types_main_metric.get(tasks_and_types.get(task_name))


def prepare_tasks() -> MTEBTasks:
    _tasks = ()
    for task_name in tasks_names:
        if task_name == "STS22":
            _tasks += (get_task(task_name, eval_splits=["test"], hf_subsets=["pl"]),)
        else:
            _tasks += (get_task(task_name, languages=["pol"]),)
    return MTEBTasks(_tasks)


class WikinewsPlClusteringS2S(AbsTaskClusteringFast):
    metadata = tasks_metadata["WikinewsPlClusteringS2S"]


class WikinewsPlClusteringP2P(AbsTaskClusteringFast):
    metadata = tasks_metadata["WikinewsPlClusteringP2P"]


class SciFieldClassification(AbsTaskClassification):
    metadata = tasks_metadata["SciField"]


class SciDefRetrieval(AbsTaskRetrieval):
    metadata = tasks_metadata["SciDefRetrieval"]


overview.TASKS_REGISTRY["WikinewsPlClusteringS2S"] = WikinewsPlClusteringS2S
overview.TASKS_REGISTRY["WikinewsPlClusteringP2P"] = WikinewsPlClusteringP2P
overview.TASKS_REGISTRY["SciField"] = SciFieldClassification
overview.TASKS_REGISTRY["SciDefRetrieval"] = SciDefRetrieval
