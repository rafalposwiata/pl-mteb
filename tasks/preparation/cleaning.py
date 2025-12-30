import re
from collections import defaultdict
from typing import List, Any
from datasets import Dataset, DatasetDict


def remove_multiple_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def remove_all_spaces(text: str) -> str:
    return text.replace(" ", "")


def normalize_text(text: str) -> str:
    return remove_all_spaces(text.strip().lower())


# Implementation based on https://gist.github.com/AlexeyVatolin/ea3adc21aa7a767603ff393b22085adc
class DatasetCleaner:

    def __init__(self, text_field: str = None, label_column: str = None, min_words: int = 3,
                 only_exact_comparisons: bool = False):
        self.text_column: str = text_field
        self.label_column: str = label_column
        self.min_words: int = min_words
        self.only_exact_comparisons: bool = only_exact_comparisons

    def clean(self, dataset: DatasetDict, text_column: str = None, validate_labels: bool = True,
              validate_leakage: bool = True, skip_splits: List = None) -> tuple[DatasetDict, dict[str, dict]]:
        if self.text_column is None:
            self.text_column = text_column
        report: dict[str, dict] = dict()
        for split in dataset.keys():
            if skip_splits is None or split not in skip_splits:
                dataset[split] = dataset[split].map(self.general_text_cleaning)
                report[split] = {"original_num_of_rows": dataset[split].num_rows}
                for func_name, func in self.get_cleaning_funcs():
                    self.execute_filter(dataset, split, report, func_name, func)

        if validate_labels:
            dataset = self.filter_unclear_label(dataset, report)

        for split in dataset.keys():
            if skip_splits is None or split not in skip_splits:
                self.execute_filter(dataset, split, report, "deduplicate_normalized",
                                    lambda ds: self.deduplicate(ds, True))

        if validate_leakage:
            for split in dataset.keys():
                if skip_splits is None or split not in skip_splits:
                    if split == "train":
                        continue
                    self.execute_filter(dataset, split, report, "leakage_from_train",
                                        lambda ds: self.filter_leakage(dataset["train"], ds))
        return dataset, report

    @staticmethod
    def execute_filter(dataset: DatasetDict, split: str, report: dict, func_name: str, func: Any) -> None:
        before = dataset[split].num_rows
        dataset[split] = func(dataset[split])
        after = dataset[split].num_rows
        report[split]["num_of_rows"] = after
        report[split][func_name] = before - after

    def general_text_cleaning(self, row):
        row[self.text_column] = remove_multiple_spaces(row[self.text_column])
        return row

    def get_cleaning_funcs(self) -> List[tuple]:
        return [
            ("empty_texts", self.filter_empty),
            ("deduplicate_exact", self.deduplicate),
            ("short_texts", self.filter_short_texts)
        ]

    def filter_empty(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda row: row[self.text_column].strip() != "")

    def deduplicate(self, dataset: Dataset, normalized_text: bool = False) -> Dataset:
        unique_texts = set()
        indices_to_keep = []
        for i, text in enumerate(dataset[self.text_column]):
            text = text.strip()
            if normalized_text:
                text = normalize_text(text)
            if text not in unique_texts:
                unique_texts.add(text)
                indices_to_keep.append(i)
        return dataset.select(indices_to_keep)

    def filter_short_texts(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda row: len(row[self.text_column].strip().split()) >= self.min_words)

    def filter_unclear_label(self, dataset: DatasetDict, report: dict) -> DatasetDict:
        all_texts = defaultdict(set)

        for ds in dataset.values():
            for text, label in zip(ds[self.text_column], ds[self.label_column]):
                all_texts[normalize_text(text)].add(label)

        texts_with_unclear_label = {t for t, labels in all_texts.items() if len(labels) > 1}
        for split in dataset.keys():
            before = dataset[split].num_rows
            dataset[split] = dataset[split].filter(
                lambda row: normalize_text(row[self.text_column]) not in texts_with_unclear_label)
            after = dataset[split].num_rows
            report[split]["num_of_rows"] = after
            report[split]["unclear_label"] = before - after
        return dataset

    def filter_leakage(self, train_dataset: Dataset, dataset: Dataset) -> Dataset:
        def normalize(text: str) -> str:
            if self.only_exact_comparisons:
                return text.strip()
            else:
                return normalize_text(text)

        train_texts = set([normalize(text) for text in train_dataset[self.text_column]])
        indices_no_leakage = [
            i for i, text in enumerate(dataset[self.text_column]) if normalize(text) not in train_texts
        ]
        return dataset.select(indices_no_leakage)
