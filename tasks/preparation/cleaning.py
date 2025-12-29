import re
from typing import Union, List
from datasets import Dataset, DatasetDict


def remove_multiple_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# Implementation based on https://gist.github.com/AlexeyVatolin/ea3adc21aa7a767603ff393b22085adc
class DatasetCleaner:

    def __init__(self, text_field: str = None, min_words: int = 3):
        self.text_column: str = text_field
        self.min_words: int = min_words

    def clean(self, dataset: DatasetDict, text_column: str = None, validate_leakage: bool = True,
              skip_splits: List = None) -> tuple[DatasetDict, dict[str, dict]]:
        if self.text_column is None:
            self.text_column = text_column
        report: dict[str, dict] = dict()
        for split in dataset.keys():
            if skip_splits is None or split not in skip_splits:
                report[split] = {"original_num_of_rows": dataset[split].num_rows}
                for func_name, func in self.get_cleaning_funcs():
                    before = dataset[split].num_rows
                    dataset[split] = func(dataset[split])
                    after = dataset[split].num_rows
                    report[split]["num_of_rows"] = after
                    report[split][func_name] = before - after

        if validate_leakage:
            for split in dataset.keys():
                if skip_splits is None or split not in skip_splits:
                    if split == "train":
                        continue
                    before = dataset[split].num_rows
                    dataset[split] = self.filter_leakage(dataset["train"], dataset[split])
                    after = dataset[split].num_rows
                    report[split]["num_of_rows"] = after
                    report[split]["leakage_from_train"] = before - after

        return dataset, report

    def get_cleaning_funcs(self) -> List[tuple]:
        return [
            ("empty_texts", self.filter_empty),
            ("deduplicate_exact", lambda _dataset: self.deduplicate(_dataset)),
            ("deduplicate_lower", lambda _dataset: self.deduplicate(_dataset, to_lower=True)),
            ("deduplicate_lower_normalized_spaces", lambda _dataset: self.deduplicate(_dataset, to_lower=True,
                                                                                      normalized_spaces=True)),
            ("short_texts", self.filter_short_texts),
        ]

    def filter_empty(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda row: row[self.text_column].strip() != "")

    def deduplicate(self, dataset: Dataset, to_lower: bool = False, normalized_spaces: bool = False) -> Dataset:
        unique_texts = set()
        indices_to_keep = []
        for i, text in enumerate(dataset[self.text_column]):
            text = text.strip()
            if to_lower:
                text = text.lower()
            if normalized_spaces:
                text = remove_multiple_spaces(text)
            if text not in unique_texts:
                unique_texts.add(text)
                indices_to_keep.append(i)
        return dataset.select(indices_to_keep)

    def filter_short_texts(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda row: len(row[self.text_column].strip().split()) >= self.min_words)

    def filter_leakage(self, train_dataset: Dataset, dataset: Dataset) -> Dataset:
        def normalize(text: str) -> str:
            return remove_multiple_spaces(text.strip().lower())

        train_texts = set([normalize(text) for text in train_dataset[self.text_column]])
        indices_no_leakage = [
            i for i, text in enumerate(dataset[self.text_column]) if normalize(text) not in train_texts
        ]
        return dataset.select(indices_no_leakage)
