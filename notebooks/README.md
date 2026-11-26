# Example notebooks

Each opens in Google Colab and installs EvalSuite in its first cell.

| Notebook | What it shows |
| --- | --- |
| [01 Classification quickstart](01_classification_quickstart.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkcs28/evalsuite-python/blob/main/notebooks/01_classification_quickstart.ipynb) | `evaluate()` with intervals, a baseline test, Markdown export, `compare()` |
| [02 Clinical TRIPOD+AI](02_clinical_tripod_ai.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkcs28/evalsuite-python/blob/main/notebooks/02_clinical_tripod_ai.ipynb) | `report_card`, diagnostic accuracy, decision curves |
| [03 LLM leaderboard](03_llm_leaderboard.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkcs28/evalsuite-python/blob/main/notebooks/03_llm_leaderboard.ipynb) | Significance-aware ranking of model runs |
| [04 Monitoring and drift](04_monitoring_drift.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkcs28/evalsuite-python/blob/main/notebooks/04_monitoring_drift.ipynb) | `Monitor`: windows, PSI / KS drift, alerts |
| [05 Vision](05_vision_segmentation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkcs28/evalsuite-python/blob/main/notebooks/05_vision_segmentation.ipynb) | Dice, IoU, HD95 |

Regenerate with `python scripts/make_notebooks.py` (`--check` runs every notebook's code).
