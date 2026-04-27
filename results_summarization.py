from typing import List
import pandas as pd
from mteb import get_model_meta
from mteb.cache import ResultCache
from mteb.models import ModelMeta
from mteb.models.model_implementations.bge_models import bge_training_data
from mteb.models.model_implementations.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.model_implementations.nvidia_models import nvidia_training_datasets
from mteb.models.model_implementations.sentence_transformers_models import sent_trf_training_dataset
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



training_datasets = {
    "sentence-transformers/distiluse-base-multilingual-cased-v2": sent_trf_training_dataset,
    "facebook/drama-base": E5_MISTRAL_TRAINING_DATA,
    "facebook/drama-large": E5_MISTRAL_TRAINING_DATA,
    "facebook/drama-1b": E5_MISTRAL_TRAINING_DATA,
    "ipipan/silver-retriever-base-v1.1": {"PolQA", "MAUPQA"},
    "sdadas/st-polish-paraphrase-from-mpnet": set(),
    "sdadas/st-polish-paraphrase-from-distilroberta": set(),

    "sdadas/mmlw-roberta-base": bge_training_data,
    "sdadas/mmlw-roberta-large": bge_training_data,
    "sdadas/mmlw-retrieval-roberta-large": bge_training_data | {"MSMARCO"},
    "sdadas/mmlw-retrieval-roberta-large-v2": nvidia_training_datasets,
    "sdadas/stella-pl": nvidia_training_datasets,
    "sdadas/stella-pl-retrieval": nvidia_training_datasets,
    "sdadas/stella-pl-retrieval-8k": nvidia_training_datasets
}


class ResultsSummarizer:

    def __init__(self, results_dir: str, models_config_path: str):
        self._tasks = prepare_tasks()
        self._models, self._models_meta = zip(*self._load_models(models_config_path))
        self._results = self._load_results(results_dir)

    def _load_results(self, results_dir: str):
        cache = ResultCache(results_dir)
        return cache.load_results(self._models, self._tasks)

    def _load_models(self, models_config_path: str) -> List[tuple[str, dict]]:
        with open(models_config_path, "r", encoding="utf-8") as file:
            return [(line.strip(), self._get_model_short_meta(line.strip())) for line in file if
                    not line.startswith("#") and line.strip() != ""]

    def create_main_table(self, table_format: str = 'psql', sort_by: str = 'Average') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        df['Average'] = self._normalize(df[tasks_names].mean(axis=1))
        for task_type in tasks.keys():
            df[task_type] = self._normalize(df[tasks[task_type]].mean(axis=1))
        df['Average (by type)'] = self._normalize(df[tasks.keys()].mean(axis=1))

        columns_with_values = list(tasks.keys()) + ['Average', 'Average (by type)']
        df = df.sort_values(sort_by)
        df = df.apply(lambda row: self._mark(row, columns_with_values,
                                             self._get_highest_values(df, columns_with_values), table_format, only_best=False), axis=1)
        for column in columns_with_values:
            df[column] = df[column].apply(self._pad)

        print('Aggregated results:')
        df["Model"] = df["Model"].apply(self._normalize_model_name)
        print(tabulate(df[['Model', 'size', "zero-shot"] + columns_with_values], headers='keys',
                       tablefmt=table_format, showindex=False))

    def crate_table_per_task_type(self, table_format: str = 'psql', sort_by: str = 'Average') -> None:
        df: pd.DataFrame = self._get_results_as_dataframe()
        for task_type in tasks.keys():
            df["zero-shot"] = df["Model_original"].apply(lambda m: self._models_meta[self._models.index(m)][f"zero-shot__{task_type}"])
            df['Average'] = self._normalize(df[tasks[task_type]].mean(axis=1))

            columns_with_values = tasks[task_type] + ['Average']
            df = df.sort_values(sort_by)
            df = df.apply(lambda row: self._mark(row, columns_with_values,
                                                 self._get_highest_values(df, columns_with_values), table_format),
                          axis=1)
            for column in columns_with_values:
                df[column] = df[column].apply(self._pad)

            print(f'Results for {task_type} task:')
            print(tabulate(df[['Model', 'zero-shot'] + columns_with_values], headers='keys',
                           tablefmt=table_format, showindex=False))

    def _get_results_as_dataframe(self) -> pd.DataFrame:
        df = self._results.to_dataframe().T
        df = df.rename(columns={i: t for i, t in enumerate(df.iloc[0].tolist())}).iloc[1:]
        df = df.astype('float64')
        for task in tasks_names:
            df[task] = df[task].apply(lambda v: self._normalize(v))
        df["Idx"] = df.index.to_series().apply(lambda m: self._models.index(m))
        df["Model"] = df.index.to_series()
        df["size"] = df["Model"].apply(lambda m: self._models_meta[self._models.index(m)]["n_parameters"])
        df["zero-shot"] = df["Model"].apply(lambda m: self._models_meta[self._models.index(m)]["zero-shot"])
        df["Model_original"] = df["Model"]
        df["Model"] = df["Model"].apply(self._normalize_model_name)
        return df

    def _get_model_short_meta(self, model_name: str) -> dict:
        meta: ModelMeta = get_model_meta(model_name)
        if model_name in training_datasets:
            meta.training_datasets = training_datasets[model_name]

        def _format(num: int) -> str:
            for unit in ["", "K", "M", "B", "T"]:
                if abs(num) < 1000:
                    if unit == "":
                        return str(int(num))
                    return f"{num:.1f}{unit}"
                num /= 1000
            return f"{num:.2f}P"

        short_meta = {"n_parameters": _format(meta.n_parameters), "zero-shot": meta.zero_shot_percentage(self._tasks)}
        for task_type in tasks.keys():
            short_meta[f"zero-shot__{task_type}"] = meta.zero_shot_percentage(prepare_tasks(task_type))

        return short_meta

    @staticmethod
    def _mark(row, columns, highest_values, table_format, only_best: bool = False):
        for column in columns:
            if row[column] == highest_values[column][0]:
                row[column] = wrap_with_marker(row[column], table_format)
            if not only_best and len(highest_values[column]) > 1 and row[column] == highest_values[column][1]:
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
    def _normalize_model_name(model_name: str) -> str:
        return model_name.split("/")[-1]

    @staticmethod
    def _pad(value):
        if isinstance(value, str):
            return value
        return "{:.2f}".format(value)


if __name__ == '__main__':
    summarizer = ResultsSummarizer('eval_results', 'configs/models.txt')
    summarizer.create_main_table(sort_by='Idx')
    summarizer.crate_table_per_task_type(sort_by='Idx')
