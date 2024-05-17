import os.path
import json
import pandas as pd
from tabulate import tabulate
from os import listdir
from models import ModelInfo
from tasks import tasks_names, tasks_types, tasks_of_type, get_main_metric, is_multilingual
from utils import from_dict


def wrap_with_marker(value, table_format, best_score: bool = True) -> str:
    if table_format == 'latex_raw':
        if best_score:
            return '\\textbf{' + str(value) + '}'
        else:
            return '\\underline{' + str(value) + '}'
    else:
        return f'{value}{"**" if best_score else "*"}'


class ResultsSummarizer:

    def __init__(self, results_dir: str, models_configs_dir: str):
        self._results = self._load_results(results_dir)
        self._models = self._load_models(models_configs_dir)

    def _load_results(self, results_dir: str):
        results = {}
        for model_name in listdir(results_dir):
            results[model_name] = {}
            for task_name in listdir(os.path.join(results_dir, model_name)):
                task_results = json.load(open(os.path.join(results_dir, model_name, task_name)))
                task_name = task_name.replace('.json', '')
                results[model_name][task_name] = self._normalize(self._get_value(task_name, task_results))
        return results

    @staticmethod
    def _load_models(models_configs_dir: str):
        models = []
        for models_config in listdir(models_configs_dir):
            with open(os.path.join(models_configs_dir, models_config), "r", encoding="utf-8") as config_file:
                models += [from_dict(ModelInfo, model_info) for model_info in json.load(config_file)]
        return models

    def create_main_table(self, table_format: str = 'psql', sort_by: str = 'Average') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        df['Average'] = self._normalize(df[tasks_names].mean(axis=1))
        for task_type in tasks_types:
            df[task_type] = self._normalize(df[tasks_of_type(task_type)].mean(axis=1))
        df['Average (by type)'] = self._normalize(df[tasks_types].mean(axis=1))

        columns_with_values = tasks_types + ['Average', 'Average (by type)']
        df = df.sort_values(sort_by)
        df = df.apply(lambda row: self._mark(row, columns_with_values,
                                             self._get_highest_values(df, columns_with_values), table_format), axis=1)
        for column in columns_with_values:
            df[column] = df[column].apply(self._pad)

        print('Aggregated results:')
        print(tabulate(df[['Model'] + columns_with_values], headers='keys',
                       tablefmt=table_format, showindex=False))

    def crate_table_per_task_type(self, table_format: str = 'psql', sort_by: str = 'Average') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        for task_type in tasks_types:
            df['Average'] = self._normalize(df[tasks_of_type(task_type)].mean(axis=1))
            df = df.sort_values(sort_by)

            columns_with_values = tasks_of_type(task_type) + ['Average']
            df = df.sort_values(sort_by)
            df = df.apply(lambda row: self._mark(row, columns_with_values,
                                                 self._get_highest_values(df, columns_with_values), table_format), axis=1)
            for column in columns_with_values:
                df[column] = df[column].apply(self._pad)

            print(f'Results for {task_type} task:')
            print(tabulate(df[['Model'] + columns_with_values], headers='keys',
                           tablefmt=table_format, showindex=False))

    def _get_results_as_dataframe(self) -> pd.DataFrame:
        model_names = [model.get_simple_name() for model in self._models]
        models_abbreviations = {model.get_simple_name(): model.get_abbreviation() for model in self._models}
        columns = ['Idx', 'Model'] + list(tasks_names)
        rows = [{**{'Idx': model_names.index(model_name) if model_name in model_names else -1,
                    'Model': models_abbreviations.get(model_name, model_name)},
                 **values_per_task}
                for model_name, values_per_task in self._results.items()]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _mark(row, columns, highest_values, table_format):
        for column in columns:
            if row[column] == highest_values[column][0]:
                row[column] = wrap_with_marker(row[column], table_format)
            if len(highest_values[column]) > 1 and row[column] == highest_values[column][1]:
                row[column] = wrap_with_marker(row[column], table_format, best_score=False)
        return row

    @staticmethod
    def _get_highest_values(df, columns):
        return {column: df[column].nlargest(2).tolist() for column in columns}

    @staticmethod
    def _get_value(task_name, task_results):
        main_metric = get_main_metric(task_name)
        if main_metric is None:
            return 0.0
        split = 'validation' if task_name == 'MSMARCO-PL' else 'test'
        result = task_results[split]
        if is_multilingual(task_name):
            result = result['pl']
        for metric_path in main_metric.split('.'):
            result = result[metric_path]
        return result

    @staticmethod
    def _normalize(value) -> float:
        v = value if isinstance(value, float) else value.tolist()[0]
        return round(100 * value if v < 1 else value, 2)

    @staticmethod
    def _pad(value):
        if isinstance(value, str):
            return value
        return "{:.2f}".format(value)


if __name__ == '__main__':
    summarizer = ResultsSummarizer('results', 'configs')
    summarizer.create_main_table(sort_by='Idx')
    summarizer.crate_table_per_task_type(sort_by='Idx')
