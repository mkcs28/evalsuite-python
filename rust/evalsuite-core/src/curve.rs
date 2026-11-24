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
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        (-score[a])
            .partial_cmp(&-score[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
    fn sum_small() {
        assert_eq!(super::np_sum(&[1.0, 2.0, 3.0]), 6.0);
    }
}
