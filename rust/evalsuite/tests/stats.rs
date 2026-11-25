mod common;

use common::*;
use evalsuite::classification::{self, binary};
use evalsuite::stats::{self as s, Adjust, Alternative, Bootstrap, BootstrapMethod, ProportionMethod, TestResult};
use serde_json::Value;

fn alt(a: &Value) -> Alternative {
    match a.as_str() {
        Some("greater") => Alternative::Greater,
        Some("less") => Alternative::Less,
        _ => Alternative::TwoSided,
    }
}

fn tr(r: evalsuite::Result<TestResult>) -> Got {
    match r {
        Ok(t) => {
            let mut v = vec![t.statistic, t.p_value, t.estimate];
            if let Some((a, b)) = t.ci {
                v.extend([a, b]);
            }
            Got::List(v)
        }
        Err(e) => Got::Err(e.to_string()),
    }
}

fn num_r(r: evalsuite::Result<f64>) -> Got {
    r.map_or_else(|e| Got::Err(e.to_string()), Got::Num)
}

fn table(v: &Value) -> Vec<Vec<f64>> {
    rows(v)
}

fn u8s(v: &Value) -> Vec<u8> {
    ints(v).into_iter().map(|x| x as u8).collect()
}

#[test]
fn stats_match_python() {
    let (n, skipped) = check("stats", 1e-9, |f, a| match f {
        "t_test" => tr(s::t_test(&nums(&a["a"]), &nums(&a["b"]), a["equal_var"].as_bool().unwrap(), alt(&a["alternative"]), 0.95)),
        "paired_t_test" => tr(s::paired_t_test(&nums(&a["a"]), &nums(&a["b"]), alt(&a["alternative"]), 0.95)),
        "mann_whitney_test" => tr(s::mann_whitney_test(&nums(&a["a"]), &nums(&a["b"]), alt(&a["alternative"]))),
        "wilcoxon_test" => tr(s::wilcoxon_test(&nums(&a["a"]), &nums(&a["b"]), alt(&a["alternative"]))),
        "kruskal_wallis_test" | "friedman_test" => {
            let g = rows(&a["groups"]);
            let refs: Vec<&[f64]> = g.iter().map(Vec::as_slice).collect();
            tr(if f == "kruskal_wallis_test" { s::kruskal_wallis_test(&refs) } else { s::friedman_test(&refs) })
        }
        "shapiro_wilk_test" => s::shapiro_wilk_test(&nums(&a["x"])).map_or_else(|e| Got::Err(e.to_string()), |(w, p)| Got::List(vec![w, p])),
        "cohens_d" => num_r(s::cohens_d(&nums(&a["a"]), &nums(&a["b"]), a["paired"].as_bool().unwrap())),
        "hedges_g" => num_r(s::hedges_g(&nums(&a["a"]), &nums(&a["b"]), a["paired"].as_bool().unwrap())),
        "cliffs_delta" => num_r(s::cliffs_delta(&nums(&a["a"]), &nums(&a["b"]))),
        "chi_square_test" => tr(s::chi_square_test(&table(&a["table"]), a["correction"].as_bool().unwrap())),
        "cramers_v" => num_r(s::cramers_v(&table(&a["table"]), a["bias_correction"].as_bool().unwrap())),
        "fisher_exact_test" => {
            let r = rows(&a["table"]);
            tr(s::fisher_exact_test([[r[0][0] as u64, r[0][1] as u64], [r[1][0] as u64, r[1][1] as u64]], alt(&a["alternative"])))
        }
        "adjust_pvalues" => {
            let m = match a["method"].as_str().unwrap() {
                "bonferroni" => Adjust::Bonferroni,
                "holm" => Adjust::Holm,
                "hochberg" => Adjust::Hochberg,
                "bh" => Adjust::Bh,
                _ => Adjust::By,
            };
            s::adjust_pvalues(&nums(&a["p"]), m).map_or_else(|e| Got::Err(e.to_string()), Got::List)
        }
        "proportion_ci" => {
            let m = match a["method"].as_str().unwrap() {
                "wilson" => ProportionMethod::Wilson,
                "clopper-pearson" => ProportionMethod::ClopperPearson,
                _ => ProportionMethod::Normal,
            };
            s::proportion_ci(a["successes"].as_u64().unwrap(), a["n"].as_u64().unwrap(), 0.9, m)
                .map_or_else(|e| Got::Err(e.to_string()), |i| Got::List(vec![i.estimate, i.low, i.high]))
        }
        "mcnemar_test" => tr(s::mcnemar_test(&ints(&a["y_true"]), &ints(&a["a"]), &ints(&a["b"]), a["exact"].as_bool(), true)),
        "roc_auc_ci" => s::roc_auc_ci(&u8s(&a["y_true"]), &nums(&a["y_prob"]), 0.95)
            .map_or_else(|e| Got::Err(e.to_string()), |i| Got::List(vec![i.estimate, i.low, i.high])),
        "delong_test" => tr(s::delong_test(&u8s(&a["y_true"]), &nums(&a["a"]), &nums(&a["b"]), 0.95)),
        "bootstrap_ci" | "paired_bootstrap_test" => {
            let y = ints(&a["y_true"]);
            let stratify = a["stratify"].as_bool().unwrap_or(false);
            let opts = Bootstrap {
                n_resamples: 300,
                seed: a["seed"].as_u64().unwrap(),
                method: match a["method"].as_str() {
                    Some("percentile") => BootstrapMethod::Percentile,
                    Some("basic") => BootstrapMethod::Basic,
                    _ => BootstrapMethod::Bca,
                },
                strata: stratify.then(|| y.iter().map(|&v| v as u64).collect()),
                ..Default::default()
            };
            let pick = |idx: &[usize], v: &[i64]| -> Vec<i64> { idx.iter().map(|&i| v[i]).collect() };
            if f == "paired_bootstrap_test" {
                let (pa, pb) = (ints(&a["a"]), ints(&a["b"]));
                return tr(s::paired_bootstrap_test(
                    y.len(),
                    |i| classification::accuracy(&pick(i, &y), &pick(i, &pa), None),
                    |i| classification::accuracy(&pick(i, &y), &pick(i, &pb), None),
                    alt(&a["alternative"]),
                    &opts,
                ));
            }
            let r = if a["metric"] == "accuracy" {
                let p = ints(&a["pred"]);
                s::bootstrap_ci(y.len(), |i| classification::accuracy(&pick(i, &y), &pick(i, &p), None), &opts)
            } else {
                let p = nums(&a["pred"]);
                s::bootstrap_ci(y.len(), |i| binary::roc_auc(&pick(i, &y), &i.iter().map(|&j| p[j]).collect::<Vec<_>>(), Some(&1), None), &opts)
            };
            r.map_or_else(|e| Got::Err(e.to_string()), |b| Got::List(vec![b.estimate, b.low, b.high]))
        }
        _ => Got::Skip,
    });
    assert!(n > 1000 && skipped == 0);
}
