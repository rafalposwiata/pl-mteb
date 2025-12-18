import logging
from tasks.preparation.tasks_preparation import BaseTask, TaskType


class AllegroReviews(BaseTask):
    def __init__(self):
        super().__init__("allegro_reviews", "allegro/klej-allegro-reviews", TaskType.CLASSIFICATION,
                         min_words=1) # Short, one-word texts such as “ok” and “super” should be retained because they express sentiment.

    def preprocess_dataset(self) -> None:
        self.rename_column("rating", "label")


class CBD(BaseTask):

    def __init__(self):
        super().__init__("cbd", "allegro/klej-cbd", TaskType.CLASSIFICATION)

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "text")
        self.rename_column("target", "label")


class PAC(BaseTask):

    def __init__(self):
        super().__init__("pac", "laugustyniak/abusive-clauses-pl", TaskType.CLASSIFICATION)


class PolEmo2In(BaseTask):

    def __init__(self):
        super().__init__("polemo2_in", "allegro/klej-polemo2-in", TaskType.CLASSIFICATION)

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "text")
        self.rename_column("target", "label")


class PolEmo2Out(BaseTask):

    def __init__(self):
        super().__init__("polemo2_out", "allegro/klej-polemo2-out", TaskType.CLASSIFICATION)

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "text")
        self.rename_column("target", "label")


class MassiveIntent(BaseTask):

    def __init__(self):
        super().__init__("massive_intent", "mteb/amazon_massive_intent", TaskType.CLASSIFICATION,
                         subset="pl", min_words=2)


class MassiveScenario(BaseTask):

    def __init__(self):
        super().__init__("massive_scenario", "mteb/amazon_massive_scenario", TaskType.CLASSIFICATION,
                         subset="pl", min_words=2)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    for task in [
        AllegroReviews,
        CBD,
        PAC,
        PolEmo2In,
        PolEmo2Out,
        MassiveIntent,
        MassiveScenario
    ]:
        _task = task()
        logging.info(f"Preparing {_task.name}")
        _task.clean()
        _task.save()
