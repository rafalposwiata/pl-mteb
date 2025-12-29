import logging
from tasks.preparation.datasets_preparation import BaseDataset, TaskType, RetrievalDataset


class MassiveIntent(BaseDataset):

    def __init__(self):
        super().__init__("massive_intent", "mteb/amazon_massive_intent", TaskType.CLASSIFICATION,
                         subset="pl", min_words=2)


class MassiveScenario(BaseDataset):

    def __init__(self):
        super().__init__("massive_scenario", "mteb/amazon_massive_scenario", TaskType.CLASSIFICATION,
                         subset="pl", min_words=2)


class ArguAna_PL(RetrievalDataset):

    def __init__(self):
        super().__init__("arguana_pl", "mteb/ArguAna-PL", queries_path="queries", corpus_path="corpus",
                         split="test", qrels_path="default")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    for task in [
        MassiveIntent,
        MassiveScenario,

        ArguAna_PL
    ]:
        _task = task()
        logging.info(f"Verifying {_task.name}")
        _task.clean()
        _task.save()
