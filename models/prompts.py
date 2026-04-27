from mteb.models.model_implementations.kalm_models import KaLM_Embedding_gemma_3_12b_task_prompts


def update_datasets_versions(_task_prompts: dict) -> dict:
    _task_prompts["CBD.v2"] = _task_prompts["CBD"]
    _task_prompts["PolEmo2.0-IN.v2"] = _task_prompts["PolEmo2.0-IN"]
    _task_prompts["PolEmo2.0-OUT.v2"] = _task_prompts["PolEmo2.0-OUT"]
    _task_prompts["AllegroReviews.v2"] = _task_prompts["AllegroReviews"]
    _task_prompts["PAC.v3"] = _task_prompts["PAC"]
    _task_prompts["SICK-E-PL.v2"] = _task_prompts["SICK-E-PL"]
    _task_prompts["CDSC-E.v2"] = _task_prompts["CDSC-E"]
    _task_prompts["PSC.v2"] = _task_prompts["PSC"]
    _task_prompts["SICK-R-PL.v2"] = _task_prompts["SICK-R-PL"]
    _task_prompts["CDSC-R.v2"] = "Retrieve semantically similar text"
    _task_prompts["EightTagsClustering.v3"] = _task_prompts["EightTagsClustering"]
    _task_prompts[
        "PlscHierarchicalClusteringS2S"] = "Identify the main and secondary category of Polish scientific articles based on the titles"
    _task_prompts[
        "PlscHierarchicalClusteringP2P"] = "Identify the main and secondary category of Polish scientific articles based on the titles and abstracts"
    _task_prompts["WikinewsPlClusteringS2S"] = "Identify the category of Wikinews articles based on the titles"
    _task_prompts[
        "WikinewsPlClusteringP2P"] = "Identify the category of Wikinews articles based on the titles and contents"
    _task_prompts["DBPedia-PLHardNegatives-query"] = _task_prompts["DBPedia-PL-query"]
    _task_prompts["HotpotQA-PLHardNegatives"] = _task_prompts["HotpotQA-PL-query"]
    _task_prompts["MSMARCO-PLHardNegatives"] = _task_prompts["MSMARCO-PL-query"]
    _task_prompts["NQ-PLHardNegatives"] = _task_prompts["NQ-PL-query"]
    _task_prompts["Quora-PLHardNegatives"] = _task_prompts["Quora-PL-query"]
    return _task_prompts


stella_prompts = {
    "Retrieval-query": "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",
    "Clustering": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "Classification": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "PairClassification": "Instruct: Retrieve semantically similar text.\nQuery: "}

bge_prompts = {
    'CBD.v2': '<instruct>Classifying the sentiment of polish tweet reviews\n<query>',
    'PolEmo2.0-IN.v2': '<instruct>Classifying the sentiment of in-domain (medicine and hotels) online reviews\n<query>',
    'PolEmo2.0-OUT.v2': '<instruct>Classifying the sentiment of out-of-domain (products and school) online reviews\n<query>',
    'AllegroReviews.v2': '<instruct>Classifying the sentiment of reviews from e-commerce marketplace Allegro\n<query>',
    'PAC.v3': '<instruct>Classifying the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"\n<query>',
    'MassiveIntentClassification': '<instruct>Given a user utterance as query, find the user intents.\n<query>',
    'MassiveScenarioClassification': '<instruct>Given a user utterance as query, find the user scenarios.\n<query>',
    'EightTagsClustering.v3': '<instruct>Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology\n<query>',
    'PlscHierarchicalClusteringS2S': '<instruct>Identify the main and secondary category of Polish scientific articles based on the titles\n<query>',
    'PlscHierarchicalClusteringP2P': '<instruct>Identify the main and secondary category of Polish scientific articles based on the titles and abstracts\n<query>',
    'WikinewsPlClusteringS2S': '<instruct>Identify the category of Wikinews articles based on the titles\n<query>',
    'WikinewsPlClusteringP2P': '<instruct>Identify the category of Wikinews articles based on the titles and contents\n<query>',
    'SICK-E-PL.v2': '<instruct>Retrieve semantically similar text.\n<query>',
    'CDSC-E.v2': '<instruct>Retrieve semantically similar text.\n<query>',
    'PSC.v2': '<instruct>Retrieve semantically similar text.\n<query>',
    'PpcPC': '<instruct>Retrieve semantically similar text.\n<query>',
    'DBPedia-PLHardNegatives-query': '<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>',
    'HotpotQA-PLHardNegatives-query': '<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>',
    'MSMARCO-PLHardNegatives-query': '<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>',
    'NQ-PLHardNegatives-query': '<instruct>Given a web search query, retrieve relevant passages that answer the query.\n<query>',
    'Quora-PLHardNegatives-query': '<instruct>Given a question, retrieve questions that are semantically equivalent to the given question.\n<query>',
    'Quora-PLHardNegatives-document': '<instruct>Given a question, retrieve questions that are semantically equivalent to the given question.\n<query>',
    'SICK-R-PL.v2': '<instruct>Retrieve semantically similar text.\n<query>',
    'CDSC-R.v2': '<instruct>Retrieve semantically similar text.\n<query>',
    'STSBenchmarkMultilingualSTS': '<instruct>Retrieve semantically similar text.\n<query>'
}

task_prompts = {
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1": update_datasets_versions(
        KaLM_Embedding_gemma_3_12b_task_prompts),
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": update_datasets_versions(KaLM_Embedding_gemma_3_12b_task_prompts),
    "Alibaba-NLP/gte-Qwen2-7B-instruct": update_datasets_versions(KaLM_Embedding_gemma_3_12b_task_prompts),
    "Qwen/Qwen3-Embedding-0.6B": update_datasets_versions(KaLM_Embedding_gemma_3_12b_task_prompts),
    "Qwen/Qwen3-Embedding-4B": update_datasets_versions(KaLM_Embedding_gemma_3_12b_task_prompts),
    "Qwen/Qwen3-Embedding-8B": update_datasets_versions(KaLM_Embedding_gemma_3_12b_task_prompts),
}

model_prompts = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": {"Retrieval-query": "query"},
    "Snowflake/snowflake-arctic-embed-m-v2.0": {"Retrieval-query": "query"},
    "facebook/drama-1b": {"Retrieval-query": "query"},
    "facebook/drama-base": {"Retrieval-query": "query"},
    "facebook/drama-large": {"Retrieval-query": "query"},
    "BAAI/bge-multilingual-gemma2": bge_prompts,

    "ipipan/silver-retriever-base-v1.1": {"Retrieval-query": "Pytanie: "},
    "sdadas/mmlw-e5-small": {"Retrieval-query": "query: ", "Retrieval-document": "passage: "},
    "sdadas/mmlw-e5-base": {"Retrieval-query": "query: ", "Retrieval-document": "passage: "},
    "sdadas/mmlw-e5-large": {"Retrieval-query": "query: ", "Retrieval-document": "passage: "},
    "sdadas/mmlw-roberta-base": {"Retrieval-query": "zapytanie: "},
    "sdadas/mmlw-roberta-large": {"Retrieval-query": "zapytanie: "},
    "sdadas/mmlw-retrieval-roberta-large": {"Retrieval-query": "zapytanie: "},
    "sdadas/mmlw-retrieval-roberta-large-v2": {"Retrieval-query": "[query]: ", "STS": "[sts]: "},
    "sdadas/stella-pl": stella_prompts,
    "sdadas/stella-pl-retrieval": stella_prompts,
    "sdadas/stella-pl-retrieval-8k": stella_prompts
}