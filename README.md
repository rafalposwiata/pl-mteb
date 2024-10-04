<h1 align="center">Polish Massive Text Embedding Benchmark </h1>
<p align="center">
    <a href="https://arxiv.org/abs/2405.10138"><img src="https://img.shields.io/badge/arXiv-2405.10138-b31b1b.svg"></a>
    <a href="https://huggingface.co/PL-MTEB"><img src="https://img.shields.io/badge/PL_MTEB-🤗-yellow"></a>
     <a href="https://huggingface.co/spaces/PL-MTEB/leaderboard"><img src="https://img.shields.io/badge/%F0%9F%8F%86-leaderboard-8A2BE2"></a>
</p>

## 💻 Installation

```bash
git clone https://github.com/rafalposwiata/pl-mteb
cd pl-mteb
pip install -r requirements.txt
```

## 🔨 Usage

```bash
python run_evaluation.py --models_config <PATH_TO_MODELS_CONFIG_FILE>

Example:
python run_evaluation.py --models_config configs/main_evaluation_configs.json
```

## 📜 Citation

```bibtex
@article{poswiata2024plmteb,
    title={PL-MTEB: Polish Massive Text Embedding Benchmark},
    author={Rafał Poświata and Sławomir Dadas and Michał Perełkiewicz},
    journal={arXiv preprint arXiv:2405.10138},
    year={2024}
}
```