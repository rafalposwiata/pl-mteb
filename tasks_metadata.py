import datasets
from mteb import TaskMetadata
import pandas as pd
from tabulate import tabulate

_MASSIVE_LANGUAGES = {
    "af": ["afr-Latn"],
    "am": ["amh-Ethi"],
    "ar": ["ara-Arab"],
    "az": ["aze-Latn"],
    "bn": ["ben-Beng"],
    "cy": ["cym-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "he": ["heb-Hebr"],
    "hi": ["hin-Deva"],
    "hu": ["hun-Latn"],
    "hy": ["hye-Armn"],
    "id": ["ind-Latn"],
    "is": ["isl-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "jv": ["jav-Latn"],
    "ka": ["kat-Geor"],
    "km": ["khm-Khmr"],
    "kn": ["kan-Knda"],
    "ko": ["kor-Kore"],
    "lv": ["lav-Latn"],
    "ml": ["mal-Mlym"],
    "mn": ["mon-Cyrl"],
    "ms": ["msa-Latn"],
    "my": ["mya-Mymr"],
    "nb": ["nob-Latn"],
    "nl": ["nld-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "sl": ["slv-Latn"],
    "sq": ["sqi-Latn"],
    "sv": ["swe-Latn"],
    "sw": ["swa-Latn"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "tl": ["tgl-Latn"],
    "tr": ["tur-Latn"],
    "ur": ["urd-Arab"],
    "vi": ["vie-Latn"],
    "zh-CN": ["cmo-Hans"],
    "zh-TW": ["cmo-Hant"],
}

_STS22_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "pl": ["pol-Latn"],
    "tr": ["tur-Latn"],
    "ar": ["ara-Arab"],
    "ru": ["rus-Cyrl"],
    "zh": ["cmn-Hans"],
    "fr": ["fra-Latn"],
    "de-en": ["deu-Latn", "eng-Latn"],
    "es-en": ["spa-Latn", "eng-Latn"],
    "it": ["ita-Latn"],
    "pl-en": ["pol-Latn", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
    "es-it": ["spa-Latn", "ita-Latn"],
    "de-fr": ["deu-Latn", "fra-Latn"],
    "de-pl": ["deu-Latn", "pol-Latn"],
    "fr-pl": ["fra-Latn", "pol-Latn"],
}

tasks_metadata = {
    "SciField": TaskMetadata(
        name="SciField",
        description="Based on the paragraph from the coursebook, identify the scientific field from which it comes. "
                    "Available fields are: chemistry, physics, mathematics, geology  and social science. #unbalanced",
        reference="https://huggingface.co/datasets/rafalposwiata/open-coursebooks-pl",
        dataset={
            "path": "PL-MTEB/scifield",
            "revision": "8064b4442da038aa0482a2542370f9a2441cc50a",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=["Academic"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"train": 5751, "test": 1438},
        avg_character_length={"train": 431.6, "test": 430.4},
    ),
    "WikinewsPlClusteringS2S": TaskMetadata(
        name="WikinewsPlClusteringS2S",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/wikinews-pl",
        dataset={
            "path": "PL-MTEB/wikinews-pl-clustering-s2s",
            "revision": "2355f412aa5b735fc66cbfe8dff766c49fb45a5d",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-06-05"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 4477},
        avg_character_length={"test": 60.18},
    ),
    "WikinewsPlClusteringP2P": TaskMetadata(
        name="WikinewsPlClusteringP2P",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/wikinews-pl",
        dataset={
            "path": "PL-MTEB/wikinews-pl-clustering-p2p",
            "revision": "8f10d8e9d094ff9f2a8e99e645ad22eb745eb3cf",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-06-05"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 4477},
        avg_character_length={"test": 1127.31},
    ),
    "SciDefRetrieval": TaskMetadata(
        name="SciDefRetrieval",
        description="Finding definitions of a word or concept from chemistry, physics, mathematics, geology and "
                    "social science coursebooks.",
        reference="https://huggingface.co/datasets/rafalposwiata/open-coursebooks-pl",
        dataset={
            "path": "PL-MTEB/scidef",
            "revision": "4755de8baebc07bed0c60cec9f796e9a346600cb",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=["Academic"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={'test': 679, 'test_corpus': 679},
        avg_character_length={'test': 52.6, 'test_corpus': 365.0},
    )
}


def generate_retrieval_stats():
    results = {}
    for task, metadata in tasks_metadata.items():
        if metadata.type == "Retrieval":
            def count_avg_length(dataset):
                return sum([len(text) for text in dataset['text']])

            queries = datasets.load_dataset(metadata.dataset["path"], "queries", ignore_verifications=True)["queries"]
            corpus = datasets.load_dataset(metadata.dataset["path"], "corpus", ignore_verifications=True)["corpus"]
            stats = {
                "test": queries.num_rows,
                "test_corpus": corpus.num_rows
            }
            len_stats = {
                "test": round(count_avg_length(queries) / queries.num_rows, 1),
                "test_corpus": round(count_avg_length(corpus) / corpus.num_rows, 1)
            }
            results[task] = f"n_samples={stats},\navg_character_length={len_stats},"
    for t, s in results.items():
        print(f'{t}\n{s}')


def get_value(field: any, field_type: any, metadata_type: str = None) -> str:
    if field is None:
        return ""
    if field_type == list:
        return ', '.join(field)
    elif field_type == dict:
        if metadata_type == 'Retrieval':
            return "\makecell[l]{q: " + str(field["test"]) + " \\\\c: " + str(field["test_corpus"]) + "}"
        elif metadata_type == 'Classification':
            return "\makecell[l]{trn: " + str(field["train"]) + " \\\\tst: " + str(field["test"]) + "}"
        else:
            return "tst: " + str(field["test"])
    else:
        return field


def prepare_task_name(task: str) -> str:
    task_name = task \
        .replace("Classification", "") \
        .replace("Clustering", "") \
        .replace("Retrieval", "")
    return "\makecell[l]{" + task_name + " \\\\  }"


def prepare_link(path: str) -> str:
    return '\\href{https://huggingface.co/datasets/' + path + '}{\\url{' + path.replace("_", "\\_") + '}} '


def generate_tasks_details_table(table_format: str = 'psql'):
    columns = ['Task', 'Link', 'Domain', 'Licence', 'No. examples', 'Avg. length']
    rows = []
    for task, metadata in tasks_metadata.items():
        print(task)
        rows.append({
            'Task': prepare_task_name(task),
            'Link': prepare_link(metadata.dataset["path"]),
            'Domain': get_value(metadata.domains, list),
            'Licence': get_value(metadata.license, str).upper(),
            'No. examples': get_value(metadata.n_samples, dict, metadata.type),
            'Avg. length': get_value(metadata.avg_character_length, dict, metadata.type)
        })

    df = pd.DataFrame(rows, columns=columns)
    tab = tabulate(df, headers='keys', tablefmt=table_format, showindex=False)
    print(tab)


if __name__ == '__main__':
    # generate_retrieval_stats()
    generate_tasks_details_table()
