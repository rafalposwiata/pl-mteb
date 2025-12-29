import json
import os
import datasets
import logging
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from datasets import DatasetDict, Dataset
from tasks.preparation.cleaning import DatasetCleaner


class TaskType(Enum):
    CLASSIFICATION = "CLASSIFICATION"
    PAIR_CLASSIFICATION = "PAIR_CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    STS = "STS"
    RETRIEVAL = "RETRIEVAL"


class AbsDataset(ABC):

    def __init__(self, name: str, task_type: TaskType):
        self.name: str = name
        self.task_type: TaskType = task_type
        self.dataset: DatasetDict = DatasetDict()
        self.output_dir: str = os.path.join("data", self.name)
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_dataset(self) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

    def save(self) -> None:
        for split, dataset in self.dataset.items():
            dataset.to_json(f"{self.output_dir}/{split}.jsonl")

    def save_json(self, filename: str, data: dict) -> None:
        with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class BaseDataset(AbsDataset):

    def __init__(self, name: str, hf_path: str, task_type: TaskType, subset: str = None, text_column: str = "text",
                 label_column: str = "label", min_words: int = 3):
        super().__init__(name, task_type)
        self.text_column: str = text_column
        self.label_column: str = label_column
        self.dataset = self.load(hf_path, subset)
        self.preprocess_dataset()
        self.cleaner = DatasetCleaner(text_column, min_words)

    @staticmethod
    def load(hf_path, subset):
        return datasets.load_dataset(hf_path, subset, verification_mode=None)

    def clean(self) -> None:
        self.dataset, cleaning_result = self.cleaner.clean(self.dataset)
        logging.info(f"Cleaning result: {cleaning_result}")
        self.save_json("cleaning_result.json", cleaning_result)

    def rename_column(self, name: str, new_name: str) -> None:
        self.dataset = self.dataset.rename_column(name, new_name)

    def class_encode_column(self, column_name: str) -> None:
        self.dataset = self.dataset.class_encode_column(column_name)


class RetrievalDataset(AbsDataset):

    def __init__(self, name: str, hf_path: str, queries_path: str, corpus_path: str, qrels_path: str,
                 split: str = None, qrels_split: str = "test", separated_qrels: bool = False, min_words: int = 3):
        super().__init__(name, TaskType.RETRIEVAL)
        self.dataset = self.load(hf_path, queries_path, corpus_path, qrels_path, split, qrels_split, separated_qrels)
        self.cleaner = DatasetCleaner(min_words=min_words)


    @staticmethod
    def load(hf_path, queries_path: str, corpus_path: str, qrels_path: str, split: str, qrels_split,
             separated_qrels: bool):
        dataset = {
            "queries": datasets.load_dataset(hf_path, name=queries_path,
                                             split=queries_path if split is None else split),
            "passages": datasets.load_dataset(hf_path, name=corpus_path, split=corpus_path if split is None else split),
            "qrels": datasets.load_dataset(hf_path + "-qrels" if separated_qrels else hf_path, name=qrels_path,
                                           split=qrels_split)
        }
        return DatasetDict(dataset)

    def clean(self) -> None:
        self.dataset, cleaning_result = self.cleaner.clean(self.dataset, text_column="text", validate_leakage=False,
                                                           skip_splits=["qrels"])
        queries_ids = set(self.dataset["queries"]["_id"])
        passages_ids = set(self.dataset["passages"]["_id"])
        self.dataset["qrels"] = self.dataset["qrels"].filter(lambda row: row["query-id"] in queries_ids
                                                                         and row["corpus-id"] in passages_ids)
        logging.info(f"Cleaning result: {cleaning_result}")
        self.save_json("cleaning_result.json", cleaning_result)
