//! COCOeval's per-image greedy matching (`evaluateImg`), for many images of one class and one area range.

/// Inputs are padded to `d_max` detections and `g_max` ground truths per image:
/// `ious[p][d][g]` (padding below every threshold), `gign[p][g]` (ignored ground truth; padding true),
/// `crowd[p][g]`, `dt_out[p][d]` (detection outside the area range), and the IoU thresholds.
/// Detections are in score order and ground truths sorted with ignored ones last, as COCOeval prepares them.
/// Returns `(dt_matched, dt_ignore)`, each `[p][t][d]` flattened.
#[allow(clippy::too_many_arguments)]
pub fn match_images(
    ious: &[f64],
    gign: &[bool],
    crowd: &[bool],
    dt_out: &[bool],
    n_images: usize,
    d_max: usize,
    g_max: usize,
    thresholds: &[f64],
) -> (Vec<bool>, Vec<bool>) {
    let t_n = thresholds.len();
    let mut dtm = vec![false; n_images * t_n * d_max];
    let mut dt_ig = vec![false; n_images * t_n * d_max];
    let mut gtm = vec![false; g_max];
    for p in 0..n_images {
        let gi = &gign[p * g_max..(p + 1) * g_max];
        let cr = &crowd[p * g_max..(p + 1) * g_max];
        for (t, &thr0) in thresholds.iter().enumerate() {
            let thr = thr0.min(1.0 - 1e-10);
            gtm.iter_mut().for_each(|x| *x = false);
            let out_base = (p * t_n + t) * d_max;
            for d in 0..d_max {
                let row = &ious[(p * d_max + d) * g_max..(p * d_max + d + 1) * g_max];
                // Best available ground truth: highest IoU at or above the threshold, the last one on ties,
                // non-ignored ones preferred; crowd regions stay available after a match.
                let mut best_reg: Option<(f64, usize)> = None;
                let mut best_any: Option<(f64, usize)> = None;
                for g in 0..g_max {
                    if gtm[g] && !cr[g] {
                        continue;
                    }
                    let v = row[g];
                    let passes = v >= thr; // false for NaN, as NumPy's comparison
                    if !passes {
                        continue;
                    }
                    if best_any.is_none_or(|(b, _)| v >= b) {
                        best_any = Some((v, g));
                    }
                    if !gi[g] && best_reg.is_none_or(|(b, _)| v >= b) {
                        best_reg = Some((v, g));
                    }
                }
                if let Some((_, m)) = best_reg.or(best_any) {
                    dtm[out_base + d] = true;
                    dt_ig[out_base + d] = gi[m];
                    gtm[m] = true;
                }
            }
            for d in 0..d_max {
                if !dtm[out_base + d] && dt_out[p * d_max + d] {
                    dt_ig[out_base + d] = true;
                }
            }
        }
    }
    (dtm, dt_ig)
}
