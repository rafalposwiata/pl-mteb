import logging
from tasks.preparation.datasets_preparation import BaseDataset, TaskType


class AllegroReviews(BaseDataset):
    def __init__(self):
        super().__init__("allegro_reviews", "allegro/klej-allegro-reviews", TaskType.CLASSIFICATION,
                         min_words=1) # Short, one-word texts such as “ok” and “super” should be retained because they express sentiment.

    def preprocess_dataset(self) -> None:
        self.rename_column("rating", "label")


class CBD(BaseDataset):

    def __init__(self):
        super().__init__("cbd", "allegro/klej-cbd", TaskType.CLASSIFICATION)

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "text")
        self.rename_column("target", "label")


class PAC(BaseDataset):

    def __init__(self):
        super().__init__("pac", "laugustyniak/abusive-clauses-pl", TaskType.CLASSIFICATION)


class PolEmo2In(BaseDataset):

    def __init__(self):
        super().__init__("polemo2_in", "allegro/klej-polemo2-in", TaskType.CLASSIFICATION)

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "text")
        self.rename_column("target", "label")


class PolEmo2Out(BaseDataset):

    def __init__(self):
        super().__init__("polemo2_out", "allegro/klej-polemo2-out", TaskType.CLASSIFICATION)

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "text")
        self.rename_column("target", "label")


class SickrPL(BaseDataset):

    def __init__(self):
        super().__init__("sickr_pl", "sdadas/sick_pl", TaskType.STS)

    def preprocess_dataset(self) -> None:
        self.remove_columns(["pair_ID", "entailment_judgment"])
        self.merge_columns(["sentence_A", "sentence_B"], "text")
        self.rename_column("sentence_A", "sentence1")
        self.rename_column("sentence_B", "sentence2")
        self.rename_column("relatedness_score", "score")

    def save(self):
        self.remove_columns(["text"])
        super().save()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    for task in [
        # AllegroReviews,
        # CBD,
        # PAC,
        # PolEmo2In,
        # PolEmo2Out,

        # SickrPL,

    ]:
        _task = task()
        logging.info(f"Preparing {_task.name}")
        _task.clean()
        _task.save()
