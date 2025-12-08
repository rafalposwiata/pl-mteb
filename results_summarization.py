from typing import List
import pandas as pd
from mteb.cache import ResultCache
from tabulate import tabulate
from tasks.tasks import tasks_names, prepare_tasks, tasks

def wrap_with_marker(value, table_format, best_score: bool = True) -> str:
    if table_format == 'latex_raw':
        if best_score:
            return '\\textbf{' + str(value) + '}'
        else:
            return '\\underline{' + str(value) + '}'
    else:
        return f'{value}{"**" if best_score else "*"}'


class ResultsSummarizer:

    def __init__(self, results_dir: str, models_config_path: str):
        self._models = self._load_models(models_config_path)
        self._tasks = prepare_tasks()
        self._results = self._load_results(results_dir)

    def _load_results(self, results_dir: str):
        cache = ResultCache(results_dir)
        return cache.load_results(self._models, self._tasks)

    @staticmethod
    def _load_models(models_config_path: str) -> List[str]:
        with open(models_config_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if not line.startswith("#") and line.strip() != ""]

    def create_main_table(self, table_format: str = 'psql', sort_by: str = 'Average') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        df['Average'] = self._normalize(df[tasks_names].mean(axis=1))
        for task_type in tasks.keys():
            df[task_type] = self._normalize(df[tasks[task_type]].mean(axis=1))
        df['Average (by type)'] = self._normalize(df[tasks.keys()].mean(axis=1))

        columns_with_values = list(tasks.keys()) + ['Average', 'Average (by type)']
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
        for task_type in tasks.keys():
            df['Average'] = self._normalize(df[tasks[task_type]].mean(axis=1))
            df = df.sort_values(sort_by)

            columns_with_values = tasks[task_type] + ['Average']
            df = df.sort_values(sort_by)
            df = df.apply(lambda row: self._mark(row, columns_with_values,
                                                 self._get_highest_values(df, columns_with_values), table_format),
                          axis=1)
            for column in columns_with_values:
                df[column] = df[column].apply(self._pad)

            print(f'Results for {task_type} task:')
            print(tabulate(df[['Model'] + columns_with_values], headers='keys',
                           tablefmt=table_format, showindex=False))

    def _get_results_as_dataframe(self) -> pd.DataFrame:
        df = self._results.to_dataframe().T
        df = df.rename(columns={i: t for i, t in enumerate(df.iloc[0].tolist())}).iloc[1:]
        df = df.astype('float64')
        df["Idx"] = df.index.to_series().apply(lambda m: self._models.index(m))
        df["Model"] = df.index.to_series()
        return df

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
    def _normalize(value) -> float:
        v = value if isinstance(value, float) else value.tolist()[0]
        return round(100 * value if v < 1 else value, 2)

    @staticmethod
    def _pad(value):
        if isinstance(value, str):
            return value
        return "{:.2f}".format(value)


if __name__ == '__main__':
    summarizer = ResultsSummarizer('eval_results', 'configs/models.txt')
    summarizer.create_main_table(sort_by='Idx')
    summarizer.crate_table_per_task_type(sort_by='Idx')
