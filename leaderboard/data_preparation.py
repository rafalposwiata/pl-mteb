import os.path
import json
from os import listdir
from models import ModelInfo
from utils import from_dict
import os
from tasks import tasks_names, tasks_types, get_main_metric, tasks, is_multilingual


class LeaderboardData:

    def __init__(self, models_config_path: str, results_dir: str):
        self.models = self._load_models(models_config_path)
        self.results = self._load_results(results_dir)

    def prepare(self) -> None:
        data_obj = {
            "title": "PL-MTEB: Polish Massive Text Embedding Benchmark",
            "description": "",
            "paper": "https://arxiv.org/abs/2405.10138",
            "citation": "@misc{powiata2024plmteb,\ntitle={PL-MTEB: Polish Massive Text Embedding Benchmark},\nauthor={Rafał Poświata and Sławomir Dadas and Michał Perełkiewicz},\nyear={2024},\neprint={2405.10138},\narchivePrefix={arXiv},\nprimaryClass={cs.CL}\n}",
            "options": {"closeable": True, "showHelp": False, "showFooter": False},
            "metrics": [{"id": "main_score", "description": "Main score"}],
            "taskGroups": [{"id": task_type, "description": ""} for task_type in tasks_types],
            "tasks": [{"id": task.name, "name": task.name, "groupId": task.task_type} for task in tasks],
            "models": [{"id": model.get_simple_name(), "name": model.model_name, "url": f"https://huggingface.co/{model.model_name}"} for model in self.models],
            "results": [{"id": model_name, "tags": [], "results": results} for model_name, results in self.results.items()]
        }
        self.save(data_obj)

    @staticmethod
    def _load_models(models_config_path: str):
        models = []

        def read_config(path):
            with open(path, "r", encoding="utf-8") as config_file:
                return [from_dict(ModelInfo, model_info) for model_info in json.load(config_file)]

        if os.path.isdir(models_config_path):
            for models_config in listdir(models_config_path):
                models += read_config(os.path.join(models_config_path, models_config))
        else:
            models += read_config(models_config_path)
        return models

    def _load_results(self, results_dir: str):
        model_names = [model.get_simple_name() for model in self.models]
        results = {}
        for model_name in listdir(results_dir):
            if model_name not in model_names:
                continue

            results[model_name] = {}
            for filename in listdir(os.path.join(results_dir, model_name)):
                task_name = filename.replace('.json', '')
                if task_name in tasks_names:
                    task_results = json.load(open(os.path.join(results_dir, model_name, filename)))
                    results[model_name][task_name] = self._flat_results(task_name, task_results)

            for task_name in tasks_names:
                if task_name not in results[model_name]:
                    results[model_name][task_name] = {"main_score": 0}

        return results

    def _flat_results(self, task_name, task_results):
        flat_results = {}
        split = 'validation' if task_name == 'MSMARCO-PL' else 'test'
        result = task_results[split]
        if is_multilingual(task_name):
            result = result['pl']

        for metric, value in result.items():
            if isinstance(value, dict):
                for sub_metric, _value in value.items():
                    flat_results[f'{metric}.{sub_metric}'] = self._normalize(_value)
            else:
                flat_results[metric] = self._normalize(value)

        flat_results["main_score"] = flat_results.get(get_main_metric(task_name), 0)
        return flat_results

    @staticmethod
    def _normalize(value) -> float:
        v = value[0] if isinstance(value, list) else value
        return round(100 * v, 5)

    @staticmethod
    def save(data) -> None:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    leaderboard_data = LeaderboardData('../configs/main_evaluation_configs.json', '../results')
    leaderboard_data.prepare()
