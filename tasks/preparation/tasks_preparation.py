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



class AbsTask(ABC):

    def __init__(self, name: str, hf_path: str, task_type: TaskType, subset: str = None):
        self.name: str = name
        self.dataset = self.load(hf_path, subset)
        self.task_type: TaskType = task_type
        self.output_dir: str = os.path.join("data", self.name)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load(hf_path, subset):
        return datasets.load_dataset(hf_path, subset, verification_mode=None)

    def preprocess_dataset(self) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

    def save(self) -> None:
        for split, dataset in self.dataset.items():
            dataset.to_json(f"{self.output_dir}/{split}.json")

    def add_idx_column(self):
        idx_column = "_idx"
        if isinstance(self.dataset, DatasetDict):
            start_idx = 1
            for split_name in self.dataset.keys():
                _dataset = self.dataset[split_name]
                end_idx = start_idx + _dataset.num_rows
                ids = [f"{i}" for i in np.arange(start_idx, end_idx, dtype=int)]
                self.dataset[split_name] = _dataset.add_column(idx_column, ids)
                start_idx = end_idx
        else:
            ids = [f"{i}" for i in np.arange(1, self.get_num_rows() + 1, dtype=int)]
            self.dataset = self.dataset.add_column(idx_column, ids)

        return idx_column

    def rename_column(self, name: str, new_name: str) -> None:
        self.dataset = self.dataset.rename_column(name, new_name)

    def class_encode_column(self, column_name: str) -> None:
        self.dataset = self.dataset.class_encode_column(column_name)

    def save_json(self, filename: str, data: dict) -> None:
        with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_num_rows(self) -> int:
        if isinstance(self.dataset, DatasetDict):
            return sum([d.num_rows for d in self.dataset.values()])
        elif isinstance(self.dataset, Dataset):
            return self.dataset.num_rows
        else:
            raise ValueError()


class BaseTask(AbsTask):

    def __init__(self, name: str, hf_path: str, task_type: TaskType, subset: str = None, text_column: str = "text",
                 label_column: str = "label", min_words: int = 3):
        super().__init__(name, hf_path, task_type, subset)
        self.text_column: str = text_column
        self.label_column: str = label_column
        self.preprocess_dataset()
        self.cleaner = DatasetCleaner(text_column, min_words)

    def clean(self) -> None:
        self.dataset, cleaning_result = self.cleaner.clean(self.dataset)
        logging.info(f"Cleaning result: {cleaning_result}")
        self.save_json("cleaning_result.json", cleaning_result)
