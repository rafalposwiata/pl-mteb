import mteb
from mteb.abstasks import AbsTaskPairClassification, AbsTaskSTS
from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.classification import AbsTaskClassification
from mteb.get_tasks import MTEBTasks, _TASKS_REGISTRY
from tasks.tasks_metadata import tasks_metadata
from typing import List

tasks: dict[str, List[str]] = {
    "Classification": [
        "CBD.v2",
        "PolEmo2.0-IN.v2",
        "PolEmo2.0-OUT.v2",
        "AllegroReviews.v2",
        "PAC.v3",
        "MassiveIntentClassification",
        "MassiveScenarioClassification"
    ],
    "Clustering": [
        "EightTagsClustering.v3",
        "PlscHierarchicalClusteringS2S",
        "PlscHierarchicalClusteringP2P",
        "WikinewsPlClusteringS2S",
        "WikinewsPlClusteringP2P"
    ],
    "PairClassification": [
        "SICK-E-PL.v2",
        "CDSC-E.v2",
        "PSC.v2",
        "PpcPC"
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
    ],
    "STS": [
        "SICK-R-PL.v2",
        "CDSC-R.v2",
        "STSBenchmarkMultilingualSTS"
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


def prepare_tasks(task_type: str = None) -> MTEBTasks:
    _tasks = ()
    for task_name in tasks_names:
        if task_type is None or task_type == tasks_and_types.get(task_name):
            if task_name == "STSBenchmarkMultilingualSTS":
                _tasks += (mteb.get_task(task_name, eval_splits=["test"], hf_subsets=["pl"]),)
            else:
                _tasks += (mteb.get_task(task_name, languages=["pol"]),)
    return MTEBTasks(_tasks)


class PacClassificationV3(AbsTaskClassification):
    metadata = tasks_metadata["PAC.v3"]


class SickePLPCV2(AbsTaskPairClassification):
    metadata = tasks_metadata["SICK-E-PL.v2"]


class CdscePCV2(AbsTaskPairClassification):
    metadata = tasks_metadata["CDSC-E.v2"]


class PscPCV2(AbsTaskPairClassification):
    metadata = tasks_metadata["PSC.v2"]


class SickrPLSTSV2(AbsTaskSTS):
    metadata = tasks_metadata["SICK-R-PL.v2"]

    min_score = 1
    max_score = 5


class CdscrSTSV2(AbsTaskSTS):
    metadata = tasks_metadata["CDSC-R.v2"]

    min_score = 1
    max_score = 5


N_SAMPLES = 2048


class EightTagsClusteringFastV2(AbsTaskClustering):
    metadata = tasks_metadata["EightTagsClustering.v3"]

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=N_SAMPLES,
        )


class WikinewsPlClusteringS2S(AbsTaskClustering):
    metadata = tasks_metadata["WikinewsPlClusteringS2S"]

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=N_SAMPLES,
        )


class WikinewsPlClusteringP2P(AbsTaskClustering):
    metadata = tasks_metadata["WikinewsPlClusteringP2P"]

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=N_SAMPLES,
        )


class PlscHierarchicalClusteringS2S(AbsTaskClustering):
    metadata = tasks_metadata["PlscHierarchicalClusteringS2S"]

    max_depth = 2

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=N_SAMPLES,
        )


class PlscHierarchicalClusteringP2P(AbsTaskClustering):
    metadata = tasks_metadata["PlscHierarchicalClusteringP2P"]

    max_depth = 2

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=N_SAMPLES,
        )


_TASKS_REGISTRY["PAC.v3"] = PacClassificationV3
_TASKS_REGISTRY["SICK-E-PL.v2"] = SickePLPCV2
_TASKS_REGISTRY["CDSC-E.v2"] = CdscePCV2
_TASKS_REGISTRY["PSC.v2"] = PscPCV2
_TASKS_REGISTRY["SICK-R-PL.v2"] = SickrPLSTSV2
_TASKS_REGISTRY["CDSC-R.v2"] = CdscrSTSV2
_TASKS_REGISTRY["EightTagsClustering.v3"] = EightTagsClusteringFastV2
_TASKS_REGISTRY["WikinewsPlClusteringS2S"] = WikinewsPlClusteringS2S
_TASKS_REGISTRY["WikinewsPlClusteringP2P"] = WikinewsPlClusteringP2P
_TASKS_REGISTRY["PlscHierarchicalClusteringS2S"] = PlscHierarchicalClusteringS2S
_TASKS_REGISTRY["PlscHierarchicalClusteringP2P"] = PlscHierarchicalClusteringP2P