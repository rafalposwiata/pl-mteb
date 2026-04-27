import mteb
import logging
from time import time
from typing import List
from transformers import HfArgumentParser
from tasks.tasks import prepare_tasks
from dataclasses import dataclass, field
from datetime import timedelta
from mteb.cache import ResultCache
from models.prompts import model_prompts, task_prompts


@dataclass
class PL_MTEBArgs:
    model: str = field(
        metadata={"help": "Path or name of model to evaluate."},
        default=None
    )
    models: str = field(
        metadata={"help": "Path to file with models to evaluate."},
        default="configs/models.txt"
    )

    def load_model_names(self) -> List[str]:
        if self.model is not None:
            return [self.model]
        else:
            with open(self.models, "r", encoding="utf-8") as file:
                return [line.strip() for line in file if not line.startswith("#") and line.strip() != ""]


class PL_MTEBEvaluator:

    def __init__(self, args: PL_MTEBArgs):
        self.args = args

    def run(self) -> None:
        for model_name in self.args.load_model_names():
            model = mteb.get_model(model_name, **self.model_kwargs(model_name))
            logging.info(f"Evaluating model: {model_name}")
            start_time = time()
            mteb.evaluate(model, prepare_tasks(), cache=ResultCache(cache_path="eval_results"),
                          encode_kwargs={"batch_size": 2})
            logging.info(f"Evaluating model {model_name} took {timedelta(seconds=time() - start_time)}.")

    def model_kwargs(self, model_name: str) -> dict:
        kwargs = {"trust_remote_code": True}
        if model_name in model_prompts:
            kwargs["model_prompts"] = model_prompts[model_name]
        if model_name in task_prompts:
            kwargs["prompts_dict"] = task_prompts[model_name]
        return kwargs


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    parser = HfArgumentParser([PL_MTEBArgs])
    args = parser.parse_args_into_dataclasses()[0]
    evaluator = PL_MTEBEvaluator(args)
    evaluator.run()
