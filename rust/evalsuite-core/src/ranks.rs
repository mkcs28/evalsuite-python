//! Mid-ranks (average ranks for ties), as `scipy.stats.rankdata(method="average")`, and the three rankings
//! DeLong's method needs from a single sort. Ranks are exact half-integers, so every result is bit-identical.

fn sorted_order(x: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    order
}

/// Average ranks (1-based) of `x`.
pub fn midranks(x: &[f64]) -> Vec<f64> {
    let order = sorted_order(x);
    let mut out = vec![0.0; x.len()];
    let mut s = 0;
    while s < order.len() {
        let mut e = s + 1;
        while e < order.len() && x[order[e]] == x[order[s]] {
            e += 1;
        }
        let r = s as f64 + (e - s + 1) as f64 / 2.0;
        for &i in &order[s..e] {
            out[i] = r;
        }
        s = e;
    }
    out
}

/// For one score column and binary labels: (tz of positives, tz of negatives, tx, ty), where tz ranks all
/// observations, tx ranks the positives among themselves and ty the negatives, each in the order the
/// positives / negatives appear in `score`.
pub fn delong_ranks(positive: &[bool], score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let order = sorted_order(score);
    // Position of each observation among the positives / negatives, in input order.
    let mut sub = vec![0usize; score.len()];
    let (mut m, mut n) = (0usize, 0usize);
    for (i, &p) in positive.iter().enumerate() {
        if p {
            sub[i] = m;
            m += 1;
        } else {
            sub[i] = n;
            n += 1;
        }
    }
    let (mut tzp, mut tzn, mut tx, mut ty) =
        (vec![0.0; m], vec![0.0; n], vec![0.0; m], vec![0.0; n]);
    let (mut s, mut sp, mut sn) = (0usize, 0usize, 0usize);
    while s < order.len() {
        let mut e = s;
        let (mut cp, mut cn) = (0usize, 0usize);
        while e < order.len() && score[order[e]] == score[order[s]] {
            if positive[order[e]] {
                cp += 1;
            } else {
                cn += 1;
            }
            e += 1;
        }
        let rz = s as f64 + (e - s + 1) as f64 / 2.0;
        let rx = sp as f64 + (cp + 1) as f64 / 2.0;
        let ry = sn as f64 + (cn + 1) as f64 / 2.0;
        for &i in &order[s..e] {
            if positive[i] {
                tzp[sub[i]] = rz;
                tx[sub[i]] = rx;
            } else {
                tzn[sub[i]] = rz;
                ty[sub[i]] = ry;
            }
        }
        sp += cp;
        sn += cn;
        s = e;
    }
    (tzp, tzn, tx, ty)
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranks() {
        assert_eq!(
            super::midranks(&[3.0, 1.0, 3.0, 2.0]),
            vec![3.5, 1.0, 3.5, 2.0]
        );
        let (tzp, tzn, tx, ty) =
            super::delong_ranks(&[true, false, true, false], &[0.9, 0.9, 0.2, 0.1]);
        assert_eq!(
            (tzp, tzn, tx, ty),
            (
                vec![3.5, 2.0],
                vec![3.5, 1.0],
                vec![2.0, 1.0],
                vec![2.0, 1.0]
            )
        );
    }
}
