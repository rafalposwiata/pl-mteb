from huggingface_hub import hf_hub_download
from huggingface_hub.repocard import metadata_load
from tasks import tasks_names
import json
from pathlib import Path


class ResultsDownloader:

    def download(self, model_name):
        readme_path = hf_hub_download(model_name, filename="README.md", etag_timeout=30)
        meta = metadata_load(readme_path)
        for sub_res in meta["model-index"][0]["results"]:
            dataset = sub_res["dataset"]
            metrics = sub_res["metrics"]
            task_type = sub_res["task"]["type"]
            split = "test"
            is_multilingual = False
            dataset_name = dataset["name"].replace("MTEB ", "")
            if "(pl)" in dataset_name:
                is_multilingual = True
                dataset_name = dataset_name.replace("(pl)", "").strip()

            if dataset_name == '8TagsClustering':
                dataset_name = 'EightTagsClustering'
            elif dataset_name == 'MSMARCO-PL':
                split = "validation"

            if dataset_name in tasks_names:
                if task_type in ['STS', 'PairClassification']:
                    metrics_values = {"cos_sim": {}, "dot": {}, "euclidean": {}, "manhattan": {}, "max": {}}
                    for metric in metrics:
                        metric_type = metric["type"]
                        parts = metric_type.split('_')
                        group = '_'.join(parts[:-1])
                        metric_name = parts[-1]
                        metrics_values[group][metric_name] = metric["value"]
                else:
                    metrics_values = {metric["type"]: metric["value"] for metric in metrics}

                results_obj = {
                    "dataset_revision": dataset["revision"],
                    "mteb_dataset_name": dataset_name,
                    "mteb_version": "1.7.49",
                    split: {"pl": metrics_values} if is_multilingual else metrics_values
                }
                self.save(model_name, dataset_name, results_obj)

    @staticmethod
    def save(model_name, task_name, data) -> None:
        dir_path = f"results/{model_name.split('/')[-1]}_generated"
        Path(dir_path).mkdir(exist_ok=True)
        with open(f'{dir_path}/{task_name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    downloader = ResultsDownloader()
    downloader.download('Alibaba-NLP/gte-Qwen2-7B-instruct')
