//! ROC AUC and average precision from one stable descending sort, bit for bit as EvalSuite's NumPy reference
//! (`_binary_curve`, `_binary_auc`, `_binary_ap` in `classification/metrics.py`).

/// NumPy's float64 `add.reduce` over a contiguous 1-D array: `pairwise_sum_DOUBLE` (8 accumulators per block
/// of up to 128, halves split on multiples of 8), added to the zero identity.
pub fn np_sum(a: &[f64]) -> f64 {
    0.0 + pairwise(a)
}

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// Summary of one binary curve: (ROC AUC, average precision, total positive weight, total negative weight).
/// AUC is NaN when either total is zero and AP when there is no positive weight (the caller raises).
pub fn curve_scores(y: &[f64], score: &[f64], w: &[f64]) -> (f64, f64, f64, f64) {
    let n = score.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, 0.0, 0.0);
    }
    // np.argsort(-score, kind="mergesort"): stable, descending; -0.0 and 0.0 compare equal.
    let order = descending_order(score);
    let mut tps = Vec::new();
    let mut fps = Vec::new();
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    for (pos, &i) in order.iter().enumerate() {
        // y * w and (1 - y) * w, accumulated sequentially as np.cumsum.
        tp += y[i] * w[i];
        fp += (1.0 - y[i]) * w[i];
        let last = pos + 1 == n;
        if last || score[i] - score[order[pos + 1]] != 0.0 {
            tps.push(tp);
            fps.push(fp);
        }
    }
    finish(&tps, &fps)
}

/// AUC and AP from the cumulative (tp, fp) at each distinct threshold, best first.
fn finish(tps: &[f64], fps: &[f64]) -> (f64, f64, f64, f64) {
    let (tp_all, fp_all) = (*tps.last().unwrap(), *fps.last().unwrap());
    let m = tps.len();
    let auc = if tp_all == 0.0 || fp_all == 0.0 {
        f64::NAN
    } else {
        // trapezoid(tpr, fpr) with fpr, tpr = r_[0, fps / fps[-1]], r_[0, tps / tps[-1]].
        let mut terms = Vec::with_capacity(m);
        let (mut x0, mut y0) = (0.0f64, 0.0f64);
        for k in 0..m {
            let (x1, y1) = (fps[k] / fp_all, tps[k] / tp_all);
            terms.push((x1 - x0) * (y1 + y0) / 2.0);
            x0 = x1;
            y0 = y1;
        }
        np_sum(&terms)
    };
    let ap = if tp_all == 0.0 {
        f64::NAN
    } else {
        // sum(diff(r_[0, rec]) * prec), rec = tps / tps[-1], prec = tps / (tps + fps).
        let mut terms = Vec::with_capacity(m);
        let mut r0 = 0.0f64;
        for k in 0..m {
            let r1 = tps[k] / tp_all;
            terms.push((r1 - r0) * (tps[k] / (tps[k] + fps[k])));
            r0 = r1;
        }
        np_sum(&terms)
    };
    (auc, ap, tp_all, fp_all)
}

/// The stable descending order of `score`, as `np.argsort(-score, kind="mergesort")`.
pub fn descending_order(score: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&a, &b| {
        (-score[a])
            .partial_cmp(&-score[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Curve scores of every bootstrap resample (`indices` is `rows × n`, row-major, each index in `0..y.len()`),
/// in parallel. With `w = None` (unit weights) the per-threshold counts are exact integers, so each resample
/// is one linear pass over the full data's presorted order with multiplicities, giving the same curve as
/// sorting the resample; with weights each resample is gathered and sorted (stable), exactly as NumPy.
pub fn bootstrap_curves(
    y: &[f64],
    score: &[f64],
    w: Option<&[f64]>,
    indices: &[i64],
    rows: usize,
) -> Vec<(f64, f64, f64, f64)> {
    use rayon::prelude::*;

    let n = indices.len().checked_div(rows).unwrap_or(0);
    if n == 0 {
        return vec![(f64::NAN, f64::NAN, 0.0, 0.0); rows];
    }
    match w {
        Some(w) => indices
            .par_chunks(n)
            .map(|idx| {
                let gy: Vec<f64> = idx.iter().map(|&i| y[i as usize]).collect();
                let gs: Vec<f64> = idx.iter().map(|&i| score[i as usize]).collect();
                let gw: Vec<f64> = idx.iter().map(|&i| w[i as usize]).collect();
                curve_scores(&gy, &gs, &gw)
            })
            .collect(),
        None => {
            let order = descending_order(score);
            indices
                .par_chunks(n)
                .map_init(
                    || vec![0u32; y.len()],
                    |counts, idx| {
                        counts.iter_mut().for_each(|c| *c = 0);
                        for &i in idx {
                            counts[i as usize] += 1;
                        }
                        let (mut tps, mut fps) = (Vec::new(), Vec::new());
                        let (mut tp, mut fp) = (0.0f64, 0.0f64);
                        let mut prev: Option<f64> = None;
                        for &i in &order {
                            let c = counts[i];
                            if c == 0 {
                                continue;
                            }
                            if let Some(p) = prev {
                                if p - score[i] != 0.0 {
                                    tps.push(tp);
                                    fps.push(fp);
                                }
                            }
                            let c = f64::from(c);
                            tp += c * y[i];
                            fp += c * (1.0 - y[i]);
                            prev = Some(score[i]);
                        }
                        tps.push(tp);
                        fps.push(fp);
                        finish(&tps, &fps)
                    },
                )
                .collect()
        }
    }
}

/// Leave-one-group-out curve scores: `group_of[i]` is the block of observation `i` (`0..groups`); result `g`
/// is the curve of every observation outside block `g`. A stable sort of a subset keeps the full data's order,
/// so one presort serves every block, with or without weights. Blocks in parallel.
pub fn leave_group_out_curves(
    y: &[f64],
    score: &[f64],
    w: Option<&[f64]>,
    group_of: &[u32],
    groups: usize,
) -> Vec<(f64, f64, f64, f64)> {
    use rayon::prelude::*;

    let order = descending_order(score);
    (0..groups)
        .into_par_iter()
        .map(|g| {
            let (mut tps, mut fps) = (Vec::new(), Vec::new());
            let (mut tp, mut fp) = (0.0f64, 0.0f64);
            let mut prev: Option<f64> = None;
            for &i in &order {
                if group_of[i] as usize == g {
                    continue;
                }
                if let Some(p) = prev {
                    if p - score[i] != 0.0 {
                        tps.push(tp);
                        fps.push(fp);
                    }
                }
                let wi = w.map_or(1.0, |w| w[i]);
                tp += y[i] * wi;
                fp += (1.0 - y[i]) * wi;
                prev = Some(score[i]);
            }
            if prev.is_none() {
                return (f64::NAN, f64::NAN, 0.0, 0.0);
            }
            tps.push(tp);
            fps.push(fp);
            finish(&tps, &fps)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn perfect_and_reversed() {
        let y = [1.0, 1.0, 0.0, 0.0];
        let w = [1.0; 4];
        let (auc, ap, tp, fp) = super::curve_scores(&y, &[0.9, 0.8, 0.2, 0.1], &w);
        assert_eq!((auc, ap, tp, fp), (1.0, 1.0, 2.0, 2.0));
        let (auc, _, _, _) = super::curve_scores(&y, &[0.1, 0.2, 0.8, 0.9], &w);
        assert_eq!(auc, 0.0);
        let (auc, _, _, _) = super::curve_scores(&y, &[0.5; 4], &w);
        assert_eq!(auc, 0.5);
    }

    #[test]
    fn bootstrap_counts_match_gathering() {
        let y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let s = [0.9, 0.9, 0.4, 0.3, 0.3, 0.1];
        let idx = [0i64, 1, 1, 3, 4, 4, 5, 2, 2, 2, 0, 3];
        let a = super::bootstrap_curves(&y, &s, None, &idx, 2);
        let b = super::bootstrap_curves(&y, &s, Some(&[1.0; 6]), &idx, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn sum_small() {
        assert_eq!(super::np_sum(&[1.0, 2.0, 3.0]), 6.0);
    }
}
