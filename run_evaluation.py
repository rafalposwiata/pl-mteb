import mteb
import logging
from time import time
from typing import List
from mteb import MTEB
from transformers import HfArgumentParser
from tasks import prepare_tasks
from dataclasses import dataclass, field
from datetime import timedelta


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
            model = mteb.get_model(model_name)
            logging.info(f"Evaluating model: {model_name}")
            start_time = time()
            evaluation = MTEB(tasks=prepare_tasks())
            evaluation.run(model, output_folder="eval_results")
            logging.info(f"Evaluating model {model_name} took {timedelta(seconds=time() - start_time)}.")


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
    logging.root.setLevel(logging.INFO)

    parser = HfArgumentParser([PL_MTEBArgs])
    args = parser.parse_args_into_dataclasses()[0]
    evaluator = PL_MTEBEvaluator(args)
    evaluator.run()
