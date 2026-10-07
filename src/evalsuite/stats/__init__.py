"""Statistical evaluation: confidence intervals, paired tests, effect sizes, multiple comparisons, compare()."""

from .compare import ComparisonResult, compare
from .effect import adjust_pvalues, cliffs_delta, cohens_d, hedges_g
from .intervals import accuracy_ci, bootstrap_ci, proportion_ci, roc_auc_ci
from .paired import delong_test, mcnemar_test, paired_bootstrap_test
from .results import ConfidenceInterval, TestResult

__all__ = [
    "ComparisonResult",
    "ConfidenceInterval",
    "TestResult",
    "accuracy_ci",
    "adjust_pvalues",
    "bootstrap_ci",
    "cliffs_delta",
    "cohens_d",
    "compare",
    "delong_test",
    "hedges_g",
    "mcnemar_test",
    "paired_bootstrap_test",
    "proportion_ci",
    "roc_auc_ci",
]
