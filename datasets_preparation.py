from datasets import Dataset, DatasetDict
import datasets


def sick_r() -> None:
    dataset = datasets.load_dataset('sdadas/sick_pl', ignore_verifications=True)
    dataset = dataset.remove_columns(['pair_ID', 'entailment_judgment'])
    dataset = dataset.rename_column("sentence_A", "sentence1")
    dataset = dataset.rename_column("sentence_B", "sentence2")
    dataset = dataset.rename_column("relatedness_score", "score")
    for split, dataset in dataset.items():
        dataset.to_json(f"sickr-pl-sts/{split}.jsonl")


def sick_e() -> None:
    dataset = datasets.load_dataset('sdadas/sick_pl', ignore_verifications=True)

    def map_label(row):
        row["labels"] = 1 if row["entailment_judgment"] == 'ENTAILMENT' else 0
        return row

    result = {}
    for split_type in dataset:
        dataset[split_type] = dataset[split_type].map(map_label)
        result[split_type] = Dataset.from_dict({
            "sent1": [dataset[split_type]["sentence_A"]],
            "sent2": [dataset[split_type]["sentence_B"]],
            "labels": [dataset[split_type]["labels"]]
        })

    for split, dataset in DatasetDict(result).items():
        dataset.to_json(f"sicke-pl-pairclassification/{split}.json")


def cdsc_r() -> None:
    dataset = datasets.load_dataset('allegro/klej-cdsc-r', ignore_verifications=True)
    dataset = dataset.remove_columns(['pair_ID'])
    dataset = dataset.rename_column("sentence_A", "sentence1")
    dataset = dataset.rename_column("sentence_B", "sentence2")
    dataset = dataset.rename_column("relatedness_score", "score")
    for split, dataset in dataset.items():
        dataset.to_json(f"cdscr-sts/{split}.jsonl")


def cdsc_e() -> None:
    dataset = datasets.load_dataset('allegro/klej-cdsc-e', ignore_verifications=True)

    def map_label(row):
        row["labels"] = 1 if row["entailment_judgment"] == 'ENTAILMENT' else 0
        return row

    result = {}
    for split_type in dataset:
        dataset[split_type] = dataset[split_type].map(map_label)
        result[split_type] = Dataset.from_dict({
            "sent1": [dataset[split_type]["sentence_A"]],
            "sent2": [dataset[split_type]["sentence_B"]],
            "labels": [dataset[split_type]["labels"]]
        })

    for split, dataset in DatasetDict(result).items():
        dataset.to_json(f"cdsce-pairclassification/{split}.json")


def ppc() -> None:
    dataset = datasets.load_dataset('sdadas/ppc', ignore_verifications=True)

    def map_label(row):
        row["labels"] = 1 if row["label"] <= 2 else 0
        return row

    result = {}
    for split_type in dataset:
        dataset[split_type] = dataset[split_type].map(map_label)
        result[split_type] = Dataset.from_dict({
            "sent1": [dataset[split_type]["sentence_A"]],
            "sent2": [dataset[split_type]["sentence_B"]],
            "labels": [dataset[split_type]["labels"]]
        })

    for split, dataset in DatasetDict(result).items():
        dataset.to_json(f"ppc-pairclassification/{split}.json")


def psc() -> None:
    dataset = datasets.load_dataset('allegro/klej-psc', ignore_verifications=True)
    result = {}
    for split_type in dataset:
        result[split_type] = Dataset.from_dict({
            "sent1": [dataset[split_type]["extract_text"]],
            "sent2": [dataset[split_type]["summary_text"]],
            "labels": [dataset[split_type]["label"]]
        })

    for split, dataset in DatasetDict(result).items():
        dataset.to_json(f"psc-pairclassification/{split}.json")


def cbd() -> None:
    dataset = datasets.load_dataset('allegro/klej-cbd', ignore_verifications=True)
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("target", "label")
    for split, dataset in dataset.items():
        dataset.to_json(f"cbd/{split}.jsonl")


def polemo2_in() -> None:
    dataset = datasets.load_dataset('allegro/klej-polemo2-in', ignore_verifications=True)
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("target", "label")
    dataset = dataset.class_encode_column("label")
    for split, dataset in dataset.items():
        dataset.to_json(f"polemo2_in/{split}.jsonl")


def polemo2_out() -> None:
    dataset = datasets.load_dataset('allegro/klej-polemo2-out', ignore_verifications=True)
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("target", "label")
    dataset = dataset.class_encode_column("label")
    for split, dataset in dataset.items():
        dataset.to_json(f"polemo2_out/{split}.jsonl")


def allegro_reviews() -> None:
    dataset = datasets.load_dataset('allegro/klej-allegro-reviews', ignore_verifications=True)
    dataset = dataset.rename_column("rating", "label")
    for split, dataset in dataset.items():
        dataset.to_json(f"allegro-reviews/{split}.jsonl")


def eight_tags() -> None:
    dataset = datasets.load_dataset('sdadas/8tags', ignore_verifications=True)

    def split(samples, n):
        for i in range(0, len(samples), n):
            yield samples[i:i + n]

    result = {}
    for split_type in dataset:
        result[split_type] = Dataset.from_dict({
            "sentences": list(split(dataset[split_type]["sentence"], 500)),
            "labels": list(split(dataset[split_type]["label"], 500))
        })

    for split, dataset in DatasetDict(result).items():
        dataset.to_json(f"8tags-clustering/{split}.jsonl")


if __name__ == '__main__':
    sick_r()
    sick_e()
    cdsc_r()
    cdsc_e()
    ppc()
    psc()
    cbd()
    polemo2_in()
    polemo2_out()
    allegro_reviews()
    eight_tags()
