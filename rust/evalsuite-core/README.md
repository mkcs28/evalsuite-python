# evalsuite-core

The compiled kernels behind [EvalSuite](https://github.com/mkcs28/evalsuite-python): edit distances, TER,
BLEU/chrF/ROUGE n-gram statistics, longest common subsequence, METEOR alignment, COCO detection matching and
bootstrap resample counting. Every kernel is tested to give exactly the result of EvalSuite's reference Python
implementation (and, through it, of sacreBLEU, rouge-score, NLTK and pycocotools).

Most users want the Python package: `pip install evalsuite-python`.

Licensed under either of MIT or Apache-2.0, at your option.
