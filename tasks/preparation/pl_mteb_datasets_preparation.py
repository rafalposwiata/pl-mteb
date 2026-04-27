import logging
import re
from datasets import Dataset
from tasks.preparation.datasets_preparation import BaseDataset, TaskType


class AllegroReviews(BaseDataset):
    def __init__(self):
        super().__init__("allegro_reviews", "allegro/klej-allegro-reviews", TaskType.CLASSIFICATION)

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
        super().__init__("sickr_pl", "sdadas/sick_pl", TaskType.STS, label_column="score")

    def preprocess_dataset(self) -> None:
        self.merge_columns(["sentence_A", "sentence_B"], "text")
        self.rename_column("sentence_A", "sentence1")
        self.rename_column("sentence_B", "sentence2")
        self.rename_column("relatedness_score", "score")

    def save(self):
        self.remove_columns(["pair_ID", "entailment_judgment", "text"])
        super().save()


class SickePL(BaseDataset):

    def __init__(self):
        super().__init__("sicke_pl", "sdadas/sick_pl", TaskType.PAIR_CLASSIFICATION, label_column="labels")

    def preprocess_dataset(self) -> None:
        self.merge_columns(["sentence_A", "sentence_B"], "text")
        self.rename_column("sentence_A", "sentence1")
        self.rename_column("sentence_B", "sentence2")

        def map_label(row):
            row["labels"] = 1 if row["entailment_judgment"] == "ENTAILMENT" else 0
            return row

        self.map(map_label)

    def save(self):
        self.remove_columns(["pair_ID", "relatedness_score", "entailment_judgment", "text"])
        for split in self.dataset.keys():
            self.dataset[split] = Dataset.from_dict(
                {column: [self.dataset[split][column]] for column in ["sentence1", "sentence2", "labels"]}
            )
        super().save()


class Cdsc_r(BaseDataset):

    def __init__(self):
        super().__init__("cdsc-r", "allegro/klej-cdsc-r", TaskType.STS, label_column="score")

    def preprocess_dataset(self) -> None:
        self.merge_columns(["sentence_A", "sentence_B"], "text")
        self.rename_column("sentence_A", "sentence1")
        self.rename_column("sentence_B", "sentence2")
        self.rename_column("relatedness_score", "score")

    def save(self):
        self.remove_columns(["pair_ID", "text"])
        super().save()


class Cdsc_e(BaseDataset):

    def __init__(self):
        super().__init__("cdsc-e", "allegro/klej-cdsc-e", TaskType.PAIR_CLASSIFICATION, label_column="labels")

    def preprocess_dataset(self) -> None:
        self.merge_columns(["sentence_A", "sentence_B"], "text")
        self.rename_column("sentence_A", "sentence1")
        self.rename_column("sentence_B", "sentence2")

        def map_label(row):
            row["labels"] = 1 if row["entailment_judgment"] == "ENTAILMENT" else 0
            return row

        self.map(map_label)

    def save(self):
        self.remove_columns(["pair_ID", "entailment_judgment", "text"])
        for split in self.dataset.keys():
            self.dataset[split] = Dataset.from_dict(
                {column: [self.dataset[split][column]] for column in ["sentence1", "sentence2", "labels"]}
            )
        super().save()


class PPC(BaseDataset):

    def __init__(self):
        super().__init__("ppc", "sdadas/ppc", TaskType.PAIR_CLASSIFICATION, label_column="labels",
                         min_words=1)

    def preprocess_dataset(self) -> None:
        self.merge_columns(["sentence_A", "sentence_B"], "text")
        self.rename_column("sentence_A", "sentence1")
        self.rename_column("sentence_B", "sentence2")

        def map_label(row):
            row["labels"] = 1 if row["label"] <= 2 else 0
            return row

        self.map(map_label)



    def save(self):
        self.remove_columns(["label", "text"])
        for split in self.dataset.keys():
            self.dataset[split] = Dataset.from_dict(
                {column: [self.dataset[split][column]] for column in ["sentence1", "sentence2", "labels"]}
            )
        super().save()


class PSC(BaseDataset):

    def __init__(self):
        super().__init__("pcs", "allegro/klej-psc", TaskType.PAIR_CLASSIFICATION, label_column="labels",
                         min_words=1)

    def preprocess_dataset(self) -> None:
        self.merge_columns(["extract_text", "summary_text"], "text")
        self.rename_column("extract_text", "sentence1")
        self.rename_column("summary_text", "sentence2")
        self.rename_column("label", "labels")

    def save(self):
        self.remove_columns(["text"])
        for split in self.dataset.keys():
            self.dataset[split] = Dataset.from_dict(
                {column: [self.dataset[split][column]] for column in ["sentence1", "sentence2", "labels"]}
            )
        super().save()



class EightTags(BaseDataset):

    def __init__(self):
        super().__init__("8tags", "sdadas/8tags", TaskType.CLUSTERING, text_column="sentences",
                         label_column="labels")

    def preprocess_dataset(self) -> None:
        self.rename_column("sentence", "sentences")
        self.rename_column("label", "labels")


class WikinewsPL(BaseDataset):

    def __init__(self, task_category: str):
        self.task_category: str = task_category
        super().__init__(f"wikinews_pl_{task_category}", "rafalposwiata/wikinews-pl", TaskType.CLUSTERING,
                         text_column="sentences", label_column="labels", validate_leakage=False)

    def preprocess_dataset(self) -> None:
        self.filter(self.filter_func)
        self.map(self.map_func)
        self.shuffle()
        self.reduce_samples(400, 500)
        self.shuffle()

    @staticmethod
    def filter_func(row) -> bool:
        if len(row["main_categories"]) > 1:
            return False
        for forbidden in ["zmarł", "zmarła", "zmarli"]:
            if forbidden in row["title"].lower():
                print(row)
                return False
        return True

    def map_func(self, row):
        def remove_starting_date(text: str) -> str:
            words = text.split(" ")
            if re.match('[\d:-]+$', words[0]) and len(words[0]) >= 10:
                return " ".join(words[1:])
            else:
                return text
        row["sentences"] = row["title"] if self.task_category == "s2s" else f"{row['title']} {row['text']}"
        row["sentences"] = remove_starting_date(row["sentences"])
        row["labels"] = row["main_categories"][0]
        return row

    def save(self):
        self.remove_columns(["url", "title", "main_categories", "categories", "text"])
        self.dataset["test"] = self.dataset["train"]
        del self.dataset["train"]
        super().save()


class WikinewsPLS2S(WikinewsPL):

    def __init__(self):
        super().__init__("s2s")


class WikinewsPLP2P(WikinewsPL):

    def __init__(self):
        super().__init__("p2p")



class Plsc(BaseDataset):

    def __init__(self, task_category: str):
        self.task_category: str = task_category
        super().__init__(f"plsc_{task_category}", "rafalposwiata/plsc", TaskType.CLUSTERING,
                         text_column="sentences", label_column="labels_merged", validate_leakage=False)

    def preprocess_dataset(self) -> None:
        self.filter(self.filter_func)
        self.map(self.map_func)
        self.shuffle()
        self.reduce_samples(50, 200)

    @staticmethod
    def filter_func(row) -> bool:
        if len(row["scientific_fields"]) > 1 or len(row["disciplines"]) > 1:
            return False
        return True

    def map_func(self, row):
        row["sentences"] = row["title"] if self.task_category == "s2s" else f"{row['title']} {row['abstract']}"
        _labels = [row["scientific_fields"][0], row["disciplines"][0]]
        row["labels"] = _labels
        row["labels_merged"] = '__'.join(_labels)
        return row

    def save(self):
        self.remove_columns(["title", "abstract", "journal", "scientific_fields", "disciplines", "labels_merged"])
        self.dataset["test"] = self.dataset["train"]
        del self.dataset["train"]
        super().save()


class PlscS2S(Plsc):

    def __init__(self):
        super().__init__("s2s")


class PlscP2P(Plsc):

    def __init__(self):
        super().__init__("p2p")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    for task in [
        # --------- Classification ---------
        AllegroReviews,
        CBD,
        PAC,
        PolEmo2In,
        PolEmo2Out,
        # --------- STS ---------
        SickrPL,
        Cdsc_r,
        # --------- Pair Classification ---------
        SickePL,
        Cdsc_e,
        PPC,
        PSC,
        # --------- Clustering ---------
        EightTags,
        WikinewsPLS2S,
        WikinewsPLP2P,
        PlscS2S,
        PlscP2P,
    ]:
        _task = task()
        logging.info(f"Preparing {_task.name}")
        _task.clean()
        _task.save()
