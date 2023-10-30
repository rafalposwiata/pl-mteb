import os.path
import json
import pandas as pd
from tabulate import tabulate
from os import listdir
from models import models_abbreviations, model_names
from tasks import tasks_names, tasks_types, tasks_of_type, get_main_metric, is_multilingual


def wrap_with_marker(value, table_format, best_score: bool = True) -> str:
    if table_format == 'latex_raw':
        if best_score:
            return '\\textbf{' + str(value) + '}'
        else:
            return '\\underline{' + str(value) + '}'
    else:
        return f'{value}{"**" if best_score else "*"}'


class ResultsSummarizer:

    def __init__(self, results_path: str):
        self._results = self._load_results(results_path)

    def _load_results(self, results_path: str):
        results = {}
        for model_name in listdir(results_path):
            results[model_name] = {}
            for task_name in listdir(os.path.join(results_path, model_name)):
                task_results = json.load(open(os.path.join(results_path, model_name, task_name)))
                task_name = task_name.replace('.json', '')
                results[model_name][task_name] = self._normalize(self._get_value(task_name, task_results))
        return results

    def create_main_table(self, table_format: str = 'psql') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        df['Average'] = self._normalize(df[tasks_names].mean(axis=1))
        for task_type in tasks_types:
            df[task_type] = self._normalize(df[tasks_of_type(task_type)].mean(axis=1))
        df['Average (by type)'] = self._normalize(df[tasks_types].mean(axis=1))

        columns_with_values = tasks_types + ['Average', 'Average (by type)']
        df = df.apply(lambda row: self._mark(row, columns_with_values,
                                             self._get_highest_values(df, columns_with_values), table_format), axis=1)
        df = df.sort_values('Idx')
        for column in columns_with_values:
            df[column] = df[column].apply(self._pad)

        print('Aggregated results:')
        print(tabulate(df[['Model'] + columns_with_values], headers='keys',
                       tablefmt=table_format, showindex=False))

    def crate_table_per_task_type(self, table_format: str = 'psql') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        for task_type in tasks_types:
            df['Average'] = self._normalize(df[tasks_of_type(task_type)].mean(axis=1))
            df = df.sort_values('Average')

            columns_with_values = tasks_of_type(task_type) + ['Average']
            df = df.apply(lambda row: self._mark(row, columns_with_values,
                                                 self._get_highest_values(df, columns_with_values), table_format), axis=1)
            df = df.sort_values('Idx')
            for column in columns_with_values:
                df[column] = df[column].apply(self._pad)

            print(f'Results for {task_type} task:')
            print(tabulate(df[['Model'] + columns_with_values], headers='keys',
                           tablefmt=table_format, showindex=False))

    def _get_results_as_dataframe(self) -> pd.DataFrame:
        columns = ['Idx', 'Model'] + list(tasks_names)
        rows = [{**{'Idx': model_names.index(model_name), 'Model': models_abbreviations.get(model_name, model_name)},
                 **values_per_task}
                for model_name, values_per_task in self._results.items()]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _mark(row, columns, highest_values, table_format):
        for column in columns:
            if row[column] == highest_values[column][0]:
                row[column] = wrap_with_marker(row[column], table_format)
            if row[column] == highest_values[column][1]:
                row[column] = wrap_with_marker(row[column], table_format, best_score=False)
        return row

    @staticmethod
    def _get_highest_values(df, columns):
        return {column: df[column].nlargest(2).tolist() for column in columns}

    @staticmethod
    def _get_value(task_name, task_results):
        main_metric = get_main_metric(task_name)
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
    summarizer = ResultsSummarizer('results')
    summarizer.create_main_table()
    summarizer.crate_table_per_task_type()
