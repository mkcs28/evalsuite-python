# evalsuite-cli

The `evalsuite` command: evaluate predictions stored in CSV, TSV, JSON or JSON-lines files, powered by the
pure-Rust [`evalsuite`](https://crates.io/crates/evalsuite) crate. Numbers match the Python package's CLI
(`pip install evalsuite-python`).

```bash
cargo install evalsuite-cli

evalsuite evaluate preds.csv --y-true label --y-pred pred --y-prob score
evalsuite evaluate housing.csv --y-true price --y-pred predicted --task regression -f json
evalsuite diagnostic screening.csv --y-true disease --y-pred test_positive
evalsuite calibration risks.csv --y-true event --y-prob risk
evalsuite text outputs.jsonl --prediction output --reference answer --reference answer2
evalsuite metrics
```

| Command | Computes |
| --- | --- |
| `evaluate` | classification (binary or multiclass, with optional probabilities) or regression, the standard metric set or `--metrics a,b,c` |
| `diagnostic` | sensitivity, specificity, PPV, NPV, LR+, LR−, DOR, Youden's J, accuracy and prevalence with 95% CIs |
| `calibration` | Brier score, ECE, MCE, calibration intercept and slope, Hosmer–Lemeshow |
| `text` | BLEU, chrF, chrF++, TER, ROUGE-1/2/L, exact match, token F1 |

Output as aligned text (default), `-f json`, `-f csv` or `-f markdown`, to standard output or `-o file`.
`-` reads CSV from standard input.

Licensed under either of MIT or Apache-2.0, at your option.
