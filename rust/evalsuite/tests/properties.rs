//! Invariants every metric must satisfy, independent of the fixtures: perfect predictions, bounds, symmetry,
//! invariance to row order and to label renaming, and agreement between equivalent formulations.

use evalsuite::classification::{self as cls, Average, Opts, ZeroDivision};
use evalsuite::{clinical, generation, judge, qa, regression, retrieval, robustness, stats};

/// A small deterministic generator (SplitMix64) so the tests need no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn labels(&mut self, n: usize, k: u64) -> Vec<i64> {
        (0..n).map(|_| self.below(k) as i64).collect()
    }
    fn normals(&mut self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                (-2.0 * self.unit().max(1e-300).ln()).sqrt()
                    * (std::f64::consts::TAU * self.unit()).cos()
            })
            .collect()
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

fn shuffle<T: Clone>(r: &mut Rng, v: &[T]) -> (Vec<T>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    for i in (1..idx.len()).rev() {
        idx.swap(i, r.below(i as u64 + 1) as usize);
    }
    (idx.iter().map(|&i| v[i].clone()).collect(), idx)
}

#[test]
fn perfect_classification_scores_one() {
    let mut r = Rng(1);
    for k in [2, 3, 7] {
        let y = r.labels(200, k);
        let o = Opts::default();
        assert_eq!(cls::accuracy(&y, &y, None).unwrap(), 1.0);
        assert_eq!(
            cls::f1(
                &y,
                &y,
                &Opts {
                    average: Average::Macro,
                    ..o.clone()
                }
            )
            .unwrap()
            .value(),
            Some(1.0)
        );
        assert_eq!(cls::mcc(&y, &y, None, ZeroDivision::Warn).unwrap(), 1.0);
        assert_eq!(
            cls::cohen_kappa(
                &y,
                &y,
                cls::KappaWeights::Unweighted,
                None,
                None,
                ZeroDivision::Warn
            )
            .unwrap(),
            1.0
        );
        assert_eq!(cls::balanced_accuracy(&y, &y, false, None).unwrap(), 1.0);
        assert_eq!(cls::hamming_loss(&y, &y, None).unwrap(), 0.0);
    }
}

#[test]
fn classification_is_row_order_invariant() {
    let mut r = Rng(2);
    for _ in 0..20 {
        let y = r.labels(300, 4);
        let p = r.labels(300, 4);
        let (ys, idx) = shuffle(&mut r, &y);
        let ps: Vec<i64> = idx.iter().map(|&i| p[i]).collect();
        let o = Opts {
            average: Average::Weighted,
            ..Opts::default()
        };
        assert!(close(
            cls::accuracy(&y, &p, None).unwrap(),
            cls::accuracy(&ys, &ps, None).unwrap()
        ));
        assert!(close(
            cls::f1(&y, &p, &o).unwrap().value().unwrap(),
            cls::f1(&ys, &ps, &o).unwrap().value().unwrap()
        ));
        assert!(close(
            cls::mcc(&y, &p, None, ZeroDivision::Warn).unwrap(),
            cls::mcc(&ys, &ps, None, ZeroDivision::Warn).unwrap()
        ));
    }
}

#[test]
fn macro_scores_do_not_depend_on_label_names() {
    let mut r = Rng(3);
    let y = r.labels(400, 3);
    let p = r.labels(400, 3);
    let rename = |v: &[i64]| -> Vec<String> {
        v.iter()
            .map(|x| ["zebra", "apple", "mango"][*x as usize].to_owned())
            .collect()
    };
    let o_i = Opts {
        average: Average::Macro,
        ..Opts::default()
    };
    let o_s = Opts {
        average: Average::Macro,
        ..Opts::default()
    };
    let a = cls::f1(&y, &p, &o_i).unwrap().value().unwrap();
    let b = cls::f1(&rename(&y), &rename(&p), &o_s)
        .unwrap()
        .value()
        .unwrap();
    assert!(close(a, b));
    assert!(close(
        cls::mcc(&y, &p, None, ZeroDivision::Warn).unwrap(),
        cls::mcc(&rename(&y), &rename(&p), None, ZeroDivision::Warn).unwrap()
    ));
}

#[test]
fn roc_auc_properties() {
    let mut r = Rng(4);
    for _ in 0..20 {
        let y: Vec<i64> = r.labels(500, 2);
        let s: Vec<f64> = r
            .normals(500)
            .iter()
            .map(|z| 1.0 / (1.0 + (-z).exp()))
            .collect();
        let auc = cls::binary::roc_auc(&y, &s, None, None).unwrap();
        assert!((0.0..=1.0).contains(&auc));
        // Flipping the scores mirrors the AUC; monotone transforms keep it.
        let neg: Vec<f64> = s.iter().map(|x| 1.0 - x).collect();
        assert!(close(
            cls::binary::roc_auc(&y, &neg, None, None).unwrap(),
            1.0 - auc
        ));
        let exp: Vec<f64> = s.iter().map(|x| x * x).collect();
        assert!(close(
            cls::binary::roc_auc(&y, &exp, None, None).unwrap(),
            auc
        ));
        // AUC equals the Mann–Whitney U statistic divided by n1·n0.
        let (pos, negs): (Vec<f64>, Vec<f64>) = (
            y.iter()
                .zip(&s)
                .filter(|(t, _)| **t == 1)
                .map(|(_, v)| *v)
                .collect(),
            y.iter()
                .zip(&s)
                .filter(|(t, _)| **t == 0)
                .map(|(_, v)| *v)
                .collect(),
        );
        let mut u = 0.0;
        for a in &pos {
            for b in &negs {
                u += if a > b {
                    1.0
                } else if a == b {
                    0.5
                } else {
                    0.0
                };
            }
        }
        assert!(close(auc, u / (pos.len() * negs.len()) as f64));
    }
}

#[test]
fn regression_identities() {
    let mut r = Rng(5);
    for _ in 0..20 {
        let y = r.normals(250);
        let p: Vec<f64> = y.iter().map(|v| v + 0.3 * r.unit() - 0.1).collect();
        let mse = regression::mse(&y, &p, None).unwrap();
        assert!(close(regression::rmse(&y, &p, None).unwrap(), mse.sqrt()));
        assert!(
            regression::mae(&y, &p, None).unwrap()
                <= regression::rmse(&y, &p, None).unwrap() + 1e-15
        );
        assert!(regression::max_error(&y, &p).unwrap() >= regression::mae(&y, &p, None).unwrap());
        assert_eq!(regression::r2(&y, &y, None).unwrap(), 1.0);
        assert_eq!(regression::mae(&y, &y, None).unwrap(), 0.0);
        // Uniform weights change nothing.
        let w = vec![2.5; y.len()];
        assert!(close(regression::mse(&y, &p, Some(&w)).unwrap(), mse));
        // Explained variance equals R² when the errors have mean zero.
        let mean_err = y.iter().zip(&p).map(|(a, b)| a - b).sum::<f64>() / y.len() as f64;
        let centred: Vec<f64> = p.iter().map(|v| v + mean_err).collect();
        assert!(
            (regression::explained_variance(&y, &centred, None).unwrap()
                - regression::r2(&y, &centred, None).unwrap())
            .abs()
                < 1e-10
        );
    }
}

#[test]
fn clinical_counts_are_consistent() {
    let mut r = Rng(6);
    for _ in 0..20 {
        let y = r.labels(300, 2);
        let p = r.labels(300, 2);
        let (tp, fp, fn_, tn) = clinical::binary_counts(&y, &p, None, None).unwrap();
        assert_eq!(tp + fp + fn_ + tn, 300.0);
        let sens = clinical::sensitivity(&y, &p, None, None, ZeroDivision::Warn).unwrap();
        let spec = clinical::specificity(&y, &p, None, None, ZeroDivision::Warn).unwrap();
        assert!(close(
            clinical::youden_j(&y, &p, None, None).unwrap(),
            sens + spec - 1.0
        ));
        assert!(close(
            clinical::lr_positive(&y, &p, None, None).unwrap(),
            sens / (1.0 - spec)
        ));
        let rep = clinical::diagnostic_report(&y, &p, None, 0.95, stats::ProportionMethod::Wilson)
            .unwrap();
        assert!(close(rep.sensitivity.estimate, sens));
        assert!(rep.sensitivity.low <= sens && sens <= rep.sensitivity.high);
    }
}

#[test]
fn intervals_cover_their_estimates() {
    for (k, n) in [(0u64, 10u64), (3, 10), (10, 10), (500, 1000), (1, 3)] {
        for m in [
            stats::ProportionMethod::Wilson,
            stats::ProportionMethod::ClopperPearson,
        ] {
            let ci = stats::proportion_ci(k, n, 0.95, m).unwrap();
            assert!(ci.low <= ci.estimate && ci.estimate <= ci.high);
            assert!((0.0..=1.0).contains(&ci.low) && (0.0..=1.0).contains(&ci.high));
            let wider = stats::proportion_ci(k, n, 0.99, m).unwrap();
            assert!(wider.low <= ci.low + 1e-15 && wider.high >= ci.high - 1e-15);
        }
    }
}

#[test]
fn p_values_are_uniform_under_the_null() {
    // Under H0 about 5% of t-tests reject at 0.05; with 2,000 tests the share is within 0.03–0.07.
    let mut r = Rng(7);
    let mut rejections = 0;
    for _ in 0..2000 {
        let a = r.normals(20);
        let b = r.normals(20);
        let t = stats::t_test(&a, &b, false, stats::Alternative::TwoSided, 0.95).unwrap();
        assert!((0.0..=1.0).contains(&t.p_value));
        rejections += usize::from(t.p_value < 0.05);
    }
    let share = rejections as f64 / 2000.0;
    assert!((0.03..0.07).contains(&share), "{share}");
}

#[test]
fn adjusted_p_values_are_monotone_and_bounded() {
    let mut r = Rng(8);
    let p: Vec<f64> = (0..50).map(|_| r.unit()).collect();
    for m in [
        stats::Adjust::Bonferroni,
        stats::Adjust::Holm,
        stats::Adjust::Bh,
        stats::Adjust::By,
        stats::Adjust::Hochberg,
    ] {
        let a = stats::adjust_pvalues(&p, m).unwrap();
        for (x, y) in p.iter().zip(&a) {
            assert!(y >= x && *y <= 1.0);
        }
    }
}

#[test]
fn text_metrics_reward_identical_outputs() {
    let refs = vec![
        vec!["the quick brown fox jumps over the lazy dog".to_owned()],
        vec!["a b c d e f".to_owned()],
    ];
    let preds = vec![
        "the quick brown fox jumps over the lazy dog".to_owned(),
        "a b c d e f".to_owned(),
    ];
    assert!(close(
        generation::bleu(&refs, &preds, &generation::BleuOpts::default()).unwrap(),
        100.0
    ));
    assert!(close(
        generation::chrf(&refs, &preds, &generation::ChrfOpts::default()).unwrap(),
        100.0
    ));
    assert_eq!(generation::ter(&refs, &preds, false).unwrap(), 0.0);
    assert!(generation::rouge_l(&refs, &preds)
        .unwrap()
        .iter()
        .all(|v| *v == 1.0));
    assert!(qa::exact_match(&refs, &preds, qa::Norm::Squad)
        .unwrap()
        .iter()
        .all(|v| *v == 1.0));
    // A worse prediction never scores higher.
    let worse = vec!["the quick fox".to_owned(), "a b".to_owned()];
    assert!(generation::bleu(&refs, &worse, &generation::BleuOpts::default()).unwrap() < 100.0);
    assert!(generation::rouge_1(&refs, &worse)
        .unwrap()
        .iter()
        .all(|v| *v < 1.0));
}

#[test]
fn retrieval_bounds_and_ideal_ranking() {
    let rel = vec![retrieval::graded([("a", 3.0), ("b", 2.0), ("c", 1.0)]).unwrap()];
    let ideal = vec![vec!["a", "b", "c", "x"]];
    let worst = vec![vec!["x", "c", "b", "a"]];
    let n = |run| {
        retrieval::ndcg_at_k(&rel, run, None, retrieval::Gains::Linear)
            .unwrap()
            .mean
    };
    assert!(close(n(&ideal), 1.0));
    assert!(n(&worst) < 1.0 && n(&worst) > 0.0);
    assert_eq!(retrieval::mrr(&rel, &ideal, None).unwrap().mean, 1.0);
    assert_eq!(retrieval::recall_at_k(&rel, &ideal, 3).unwrap().mean, 1.0);
}

#[test]
fn judge_statistics() {
    // Bradley–Terry recovers the order of a consistent tournament; strengths are centred.
    let mut comps = Vec::new();
    for (a, b, wins_a, n) in [("A", "B", 8, 10), ("B", "C", 7, 10), ("A", "C", 9, 10)] {
        for i in 0..n {
            comps.push((a, b, if i < wins_a { 1.0 } else { 0.0 }));
        }
    }
    let r = judge::bradley_terry(&comps, judge::BtScale::Log, 10_000, 1e-12).unwrap();
    let s: f64 = r.values.iter().sum();
    assert!(s.abs() < 1e-9);
    let pos = |m: &str| r.models.iter().position(|x| *x == m).unwrap();
    assert!(r.values[pos("A")] > r.values[pos("B")] && r.values[pos("B")] > r.values[pos("C")]);
    // Perfect agreement gives alpha = 1.
    let ratings = vec![vec![1.0, 2.0, 3.0, 2.0], vec![1.0, 2.0, 3.0, 2.0]];
    assert!(close(
        judge::krippendorff_alpha(&ratings, judge::Level::Interval).unwrap(),
        1.0
    ));
}

#[test]
fn robustness_drop_identities() {
    let mut r = Rng(9);
    let c: Vec<bool> = (0..200).map(|_| r.unit() < 0.8).collect();
    let p: Vec<bool> = c.iter().map(|&x| x && r.unit() < 0.9).collect();
    let d = robustness::noise_robustness(&c, &p).unwrap();
    assert!(close(d.absolute_drop, d.clean_accuracy - d.value));
    assert!(d.flip_rate >= 0.0 && d.flip_rate <= 1.0);
    let same = robustness::noise_robustness(&c, &c).unwrap();
    assert_eq!(same.absolute_drop, 0.0);
    assert_eq!(same.flip_rate, 0.0);
}

#[test]
fn bootstrap_is_reproducible_and_brackets_the_estimate() {
    let mut r = Rng(10);
    let x = r.normals(300);
    let mean = |idx: &[usize]| Ok(idx.iter().map(|&i| x[i]).sum::<f64>() / idx.len() as f64);
    let opts = stats::Bootstrap {
        n_resamples: 500,
        seed: 42,
        ..Default::default()
    };
    let a = stats::bootstrap_ci(x.len(), &mean, &opts).unwrap();
    let b = stats::bootstrap_ci(x.len(), &mean, &opts).unwrap();
    assert_eq!(a, b);
    assert!(a.low <= a.estimate && a.estimate <= a.high);
}

#[test]
fn errors_on_invalid_input() {
    assert!(cls::accuracy::<i64>(&[], &[], None).is_err());
    assert!(cls::accuracy(&[1, 0], &[1], None).is_err());
    assert!(regression::mae(&[1.0, f64::NAN], &[1.0, 2.0], None).is_err());
    assert!(cls::binary::roc_auc(&[1, 1, 1], &[0.1, 0.2, 0.3], None, None).is_err());
    assert!(stats::proportion_ci(5, 0, 0.95, stats::ProportionMethod::Wilson).is_err());
    assert!(
        generation::bleu::<String, String>(&[], &[], &generation::BleuOpts::default()).is_err()
    );
}
