from mteb import TaskMetadata

tasks_metadata = {
    "PAC.v3": TaskMetadata(
        name="PAC.v3",
        description="Polish Paraphrase Corpus This version corrects errors found in the original data.",
        reference="https://proceedings.neurips.cc/paper_files/paper/2022/file/890b206ebb79e550f3988cb8db936f42-Paper-Datasets_and_Benchmarks.pdf",
        dataset={
            "path": "PL-MTEB/pac",
            "revision": "bd528c93a97d6140a8cdffb8ebd29a5c7b77fa58",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),  # best guess: based on publication date
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{NEURIPS2022_890b206e,
 author = {Augustyniak, Lukasz and Tagowski, Kamil and Sawczyn, Albert and Janiak, Denis and Bartusiak, Roman and Szymczak, Adrian and Janz, Arkadiusz and Szyma\'{n}ski, Piotr and W\k{a}troba, Marcin and Morzy, Miko\l aj and Kajdanowicz, Tomasz and Piasecki, Maciej},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {21805--21818},
 publisher = {Curran Associates, Inc.},
 title = {This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/890b206ebb79e550f3988cb8db936f42-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}
""",
        adapted_from=["PacClassification"],
    ),
    "SICK-E-PL.v2": TaskMetadata(
        name="SICK-E-PL.v2",
        dataset={
            "path": "PL-MTEB/sicke-pl-pairclassification",
            "revision": "4cf810083275ec99e9945b2d55b22fb7c3a6fb06",
        },
        description="Polish version of SICK dataset for textual entailment.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Reviews"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{dadas-etal-2020-evaluation,
  address = {Marseille, France},
  author = {Dadas, Slawomir  and
Pere{\l}kiewicz, Micha{\l}  and
Po{\'s}wiata, Rafa{\l}},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\'e}l{\`e}ne  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {1674--1680},
  publisher = {European Language Resources Association},
  title = {Evaluation of Sentence Representations in {P}olish},
  url = {https://aclanthology.org/2020.lrec-1.207},
  year = {2020},
}
""",
        adapted_from=["SICK-E-PL"]
    ),
    "CDSC-E.v2": TaskMetadata(
        name="CDSC-E.v2",
        dataset={
            "path": "PL-MTEB/cdsce-pairclassification",
            "revision": "d801ce08d97cd03568388dc9449ba5bb652b4663",
        },
        description="Compositional Distributional Semantics Corpus for textual entailment.",
        reference="https://aclanthology.org/P17-1073.pdf",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wroblewska-krasnowska-kieras-2017-polish,
  address = {Vancouver, Canada},
  author = {Wr{\'o}blewska, Alina  and
Krasnowska-Kiera{\'s}, Katarzyna},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/P17-1073},
  editor = {Barzilay, Regina  and
Kan, Min-Yen},
  month = jul,
  pages = {784--792},
  publisher = {Association for Computational Linguistics},
  title = {{P}olish evaluation dataset for compositional distributional semantics models},
  url = {https://aclanthology.org/P17-1073},
  year = {2017},
}
""",
        adapted_from=["CDSC-E"]
    ),
    "PSC.v2": TaskMetadata(
        name="PSC.v2",
        dataset={
            "path": "PL-MTEB/psc-pairclassification",
            "revision": "9fedecda0daab230707afdd92d8d7074c637c1dd",
        },
        description="Polish Summaries Corpus",
        reference="http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",  # and https://zil.ipipan.waw.pl/PolishSummariesCorpus
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="max_ap",
        date=("1996-01-01", "2003-01-01"),  # from the paper
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ogrodniczuk-kopec-2014-polish,
  address = {Reykjavik, Iceland},
  author = {Ogrodniczuk, Maciej  and
Kope{\'c}, Mateusz},
  booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Declerck, Thierry  and
Loftsson, Hrafn  and
Maegaard, Bente  and
Mariani, Joseph  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  month = may,
  pages = {3712--3715},
  publisher = {European Language Resources Association (ELRA)},
  title = {The {P}olish Summaries Corpus},
  url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf},
  year = {2014},
}
""",
        adapted_from=["PSC"]
    ),
    "SICK-R-PL.v2": TaskMetadata(
        name="SICK-R-PL.v2",
        dataset={
            "path": "PL-MTEB/sickr-pl-sts",
            "revision": "fd5c2441b7eeff8676768036142af4cfa42c1339",
        },
        description="Polish version of SICK dataset for textual relatedness.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="cosine_spearman",
        date=("2018-01-01", "2019-09-01"),  # rough estimate
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@inproceedings{dadas-etal-2020-evaluation,
  address = {Marseille, France},
  author = {Dadas, Slawomir  and
Perelkiewicz, Michal  and
Poswiata, Rafal},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, Helene  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {1674--1680},
  publisher = {European Language Resources Association},
  title = {Evaluation of Sentence Representations in {P}olish},
  url = {https://aclanthology.org/2020.lrec-1.207},
  year = {2020},
}
""",
        adapted_from=["SICK-R-PL"]
    ),
    "CDSC-R.v2": TaskMetadata(
        name="CDSC-R.v2",
        dataset={
            "path": "PL-MTEB/cdscr-sts",
            "revision": "1cd6abbb00df7d14be3dbd76a7dcc64b3a79a7cd",
        },
        description="Compositional Distributional Semantics Corpus for textual relatedness.",
        reference="https://aclanthology.org/P17-1073.pdf",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2017-04-01"),  # rough estimate
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@inproceedings{wroblewska-krasnowska-kieras-2017-polish,
  address = {Vancouver, Canada},
  author = {Wr{\'o}blewska, Alina  and
Krasnowska-Kiera{\'s}, Katarzyna},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/P17-1073},
  editor = {Barzilay, Regina  and
Kan, Min-Yen},
  month = jul,
  pages = {784--792},
  publisher = {Association for Computational Linguistics},
  title = {{P}olish evaluation dataset for compositional distributional semantics models},
  url = {https://aclanthology.org/P17-1073},
  year = {2017},
}
""",
        adapted_from=["SICK-R-PL"]
    ),
    "EightTagsClustering.v3": TaskMetadata(
        name="EightTagsClustering.v3",
        description="Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, "
        + "food, medicine, motorization, work, sport and technology.",
        reference="https://aclanthology.org/2020.lrec-1.207.pdf",
        dataset={
            "path": "PL-MTEB/8tags-clustering",
            "revision": "a0b2efa8af06a5ada2f1a84608e8b2b155d616b2",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2019-01-01", "2020-05-01"),
        domains=["Social", "Written"],
        task_subtypes=["Topic classification", "Thematic clustering"],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dadas-etal-2020-evaluation,
  address = {Marseille, France},
  author = {Dadas, Slawomir  and
Pere{\\l}kiewicz, Micha{\\l}  and
Po{\\'s}wiata, Rafa{\\l}},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\\'e}chet, Fr{\\'e}d{\\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\\'e}l{\\`e}ne  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  isbn = {979-10-95546-34-4},
  language = {English},
  month = may,
  pages = {1674--1680},
  publisher = {European Language Resources Association},
  title = {Evaluation of Sentence Representations in {P}olish},
  url = {https://aclanthology.org/2020.lrec-1.207},
  year = {2020},
}
""",
        adapted_from=["EightTagsClustering.v2"],
    ),
    "WikinewsPlClusteringS2S": TaskMetadata(
        name="WikinewsPlClusteringS2S",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/wikinews-pl",
        dataset={
            "path": "PL-MTEB/wikinews-pl-clustering-s2s",
            "revision": "9c1df563a3592404f23cca8f8ed17def991f05b7",
        },
        type="Clustering",
        category="t2c",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-06-05"),
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation=""
    ),
    "WikinewsPlClusteringP2P": TaskMetadata(
        name="WikinewsPlClusteringP2P",
        description="",
        reference="https://huggingface.co/datasets/rafalposwiata/wikinews-pl",
        dataset={
            "path": "PL-MTEB/wikinews-pl-clustering-p2p",
            "revision": "964eb753157e7b0ab5e96b42888a714004560ecc",
        },
        type="Clustering",
        category="t2c",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2024-06-05"),
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
    ),
    "PlscHierarchicalClusteringS2S": TaskMetadata(
        name="PlscHierarchicalClusteringS2S",
        description="Clustering of Polish article titles from Library of Science (https://bibliotekanauki.pl/), either "
        + "on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "PL-MTEB/plsc-hierarchical-clustering-s2s",
            "revision": "7f192062f3db1c0e5020e38d0382ea1613a427ab",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        domains=["Academic", "Written"],
        task_subtypes=["Topic classification", "Thematic clustering"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        adapted_from=["PlscClusteringS2S.v2"],
    ),
    "PlscHierarchicalClusteringP2P": TaskMetadata(
        name="PlscHierarchicalClusteringP2P",
        description="Clustering of Polish article titles from Library of Science (https://bibliotekanauki.pl/), either "
        + "on the scientific field or discipline.",
        reference="https://huggingface.co/datasets/rafalposwiata/plsc",
        dataset={
            "path": "PL-MTEB/plsc-hierarchical-clustering-p2p",
            "revision": "9a6f79c41a8ef71ee60a4b79f4030542ee354237",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="v_measure",
        date=("2022-04-04", "2023-09-12"),
        domains=["Academic", "Written"],
        task_subtypes=["Topic classification", "Thematic clustering"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        adapted_from=["PlscClusteringP2P.v2"],
    )
}
