from typing import List
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from models import ModelInfo, models, Model, RetrievalModel
from tasks import TaskInfo, tasks, new_tasks


class PlMtebEvaluator:

    def __init__(self, models: List[ModelInfo], tasks: List[TaskInfo]):
        self._models: List[ModelInfo] = models
        self._tasks: List[TaskInfo] = tasks

    def run(self) -> None:
        for model_info in self._models:
            print(f'Evaluated model: {model_info.model_name}')
            for task_info in self._tasks:
                model = self._create_model(model_info, task_info)
                task = self._get_task(task_info)
                eval_splits = ['validation'] if task_info.name == 'MSMARCO-PL' else ['test']
                evaluation = MTEB(tasks=[task], task_langs=["pl"])
                evaluation.run(model,
                               eval_splits=eval_splits,
                               output_folder=f"results/{model_info.get_simple_name()}")

    @staticmethod
    def _create_model(model_info: ModelInfo, task_info: TaskInfo):
        model = SentenceTransformer(model_info.model_name)
        model.eval()
        if model_info.fp16:
            model.half()
        return RetrievalModel(model, model_info) if task_info.task_type == 'Retrieval' else Model(model, model_info)

    @staticmethod
    def _get_task(task_info: TaskInfo):
        task_name = task_info.name
        if task_name in new_tasks:
            return new_tasks.get(task_name)
        return task_name


if __name__ == '__main__':
    evaluator = PlMtebEvaluator(models=models, tasks=tasks)
    evaluator.run()
