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

/// Detections and ground truth of every image, concatenated: image `i` owns rows `off[i]..off[i + 1]`.
/// Boxes are `[x1, y1, x2, y2]` rows (validated finite by the caller); scores are finite.
pub struct Images<'a> {
    pub gt_boxes: &'a [f64],
    pub gt_labels: &'a [i64],
    pub gt_crowd: &'a [bool],
    pub gt_area: &'a [f64],
    pub gt_off: &'a [usize],
    pub dt_boxes: &'a [f64],
    pub dt_labels: &'a [i64],
    pub dt_scores: &'a [f64],
    pub dt_off: &'a [usize],
}

/// COCO evaluation result: the classes in ascending order, `precision[t][r][k][a][m]` and `recall[t][k][a][m]`
/// (row-major), `-1` where a class has no ground truth to score.
pub struct Evaluation {
    pub cats: Vec<i64>,
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
}

/// One image and one class: ground truth (original order) and score-sorted, truncated detections with IoUs.
struct Pair {
    gt_crowd: Vec<bool>,
    gt_area: Vec<f64>,
    dt_scores: Vec<f64>,
    dt_area: Vec<f64>,
    ious: Vec<f64>, // [d][g]
    n_gt: usize,
}

fn area(b: &[f64]) -> f64 {
    (b[2] - b[0]) * (b[3] - b[1])
}

/// IoU of a detection with a ground truth box, with every operation in the order of EvalSuite's NumPy
/// `_iou_matrix` (and pycocotools): for crowd ground truth the union is the detection's area.
fn iou(d: &[f64], g: &[f64], a_d: f64, a_g: f64, crowd: bool) -> f64 {
    let w = (d[2].min(g[2]) - d[0].max(g[0])).max(0.0);
    let h = (d[3].min(g[3]) - d[1].max(g[1])).max(0.0);
    let inter = w * h;
    let union = if crowd { a_d } else { a_d + a_g - inter };
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

fn stable_desc(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // Stable sort of the negated scores, as np.argsort(-scores, kind="mergesort"): ties (including +0/-0)
    // keep their original order. Scores are finite, so partial_cmp never fails.
    order.sort_by(|&a, &b| {
        (-scores[a])
            .partial_cmp(&-scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Full COCO bbox evaluation (COCOeval.evaluate + accumulate), parallel over classes. Same results as
/// EvalSuite's Python implementation (and pycocotools).
pub fn evaluate(
    img: &Images<'_>,
    iou_thrs: &[f64],
    max_dets: &[usize],
    areas: &[(f64, f64)],
    rec_thrs: &[f64],
) -> Evaluation {
    use rayon::prelude::*;

    let n_img = img.gt_off.len() - 1;
    let mut cats: Vec<i64> = img
        .gt_labels
        .iter()
        .chain(img.dt_labels.iter())
        .copied()
        .collect();
    cats.sort_unstable();
    cats.dedup();
    let k_n = cats.len();
    let top = max_dets.iter().copied().max().unwrap_or(0);
    // Group each image's boxes by class once: by_cat[k] lists (gt rows, dt rows) of every image that has
    // class k, in image order.
    let mut by_cat: Vec<Vec<(Vec<usize>, Vec<usize>)>> = vec![Vec::new(); k_n];
    let cat_index = |c: i64| cats.binary_search(&c).expect("class present");
    for i in 0..n_img {
        let mut local: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
        let slot = |k: usize, local: &mut Vec<(usize, Vec<usize>, Vec<usize>)>| -> usize {
            match local.iter().position(|e| e.0 == k) {
                Some(p) => p,
                None => {
                    local.push((k, Vec::new(), Vec::new()));
                    local.len() - 1
                }
            }
        };
        for g in img.gt_off[i]..img.gt_off[i + 1] {
            let s = slot(cat_index(img.gt_labels[g]), &mut local);
            local[s].1.push(g);
        }
        for d in img.dt_off[i]..img.dt_off[i + 1] {
            let s = slot(cat_index(img.dt_labels[d]), &mut local);
            local[s].2.push(d);
        }
        for (k, g, d) in local {
            by_cat[k].push((g, d));
        }
    }
    let (t_n, r_n, a_n, m_n) = (iou_thrs.len(), rec_thrs.len(), areas.len(), max_dets.len());
    let per_cat: Vec<(Vec<f64>, Vec<f64>)> = by_cat
        .par_iter()
        .map(|entries| {
            let pairs: Vec<Pair> = entries
                .iter()
                .map(|(gs, ds)| {
                    let scores: Vec<f64> = ds.iter().map(|&d| img.dt_scores[d]).collect();
                    let mut order = stable_desc(&scores);
                    order.truncate(top);
                    let dts: Vec<usize> = order.iter().map(|&o| ds[o]).collect();
                    let dt_area: Vec<f64> = dts
                        .iter()
                        .map(|&d| area(&img.dt_boxes[4 * d..4 * d + 4]))
                        .collect();
                    let mut ious = Vec::with_capacity(dts.len() * gs.len());
                    for (di, &d) in dts.iter().enumerate() {
                        for &g in gs {
                            ious.push(iou(
                                &img.dt_boxes[4 * d..4 * d + 4],
                                &img.gt_boxes[4 * g..4 * g + 4],
                                dt_area[di],
                                area(&img.gt_boxes[4 * g..4 * g + 4]),
                                img.gt_crowd[g],
                            ));
                        }
                    }
                    Pair {
                        gt_crowd: gs.iter().map(|&g| img.gt_crowd[g]).collect(),
                        gt_area: gs.iter().map(|&g| img.gt_area[g]).collect(),
                        dt_scores: dts.iter().map(|&d| img.dt_scores[d]).collect(),
                        dt_area,
                        ious,
                        n_gt: gs.len(),
                    }
                })
                .collect();
            let mut prec = vec![-1.0f64; t_n * r_n * a_n * m_n]; // [t][r][a][m]
            let mut rec = vec![-1.0f64; t_n * a_n * m_n]; // [t][a][m]
            for (a, &(lo, hi)) in areas.iter().enumerate() {
                // Match every image: per pair, dt_matched / dt_ignore [t][d] and the non-ignored ground truth count.
                let mut npig = 0usize;
                let mut evals: Vec<(Vec<bool>, Vec<bool>)> = Vec::with_capacity(pairs.len());
                for p in &pairs {
                    let g_ignore: Vec<bool> = (0..p.n_gt)
                        .map(|g| p.gt_crowd[g] || p.gt_area[g] < lo || p.gt_area[g] > hi)
                        .collect();
                    // Ground truth with ignored boxes last, otherwise in original order (stable).
                    let mut gorder: Vec<usize> = (0..p.n_gt).collect();
                    gorder.sort_by_key(|&g| g_ignore[g]);
                    npig += g_ignore.iter().filter(|&&x| !x).count();
                    let (d_n, g_n) = (p.dt_scores.len(), p.n_gt);
                    let gi: Vec<bool> = gorder.iter().map(|&g| g_ignore[g]).collect();
                    let cr: Vec<bool> = gorder.iter().map(|&g| p.gt_crowd[g]).collect();
                    let mut sorted_ious = vec![0.0f64; d_n * g_n];
                    for d in 0..d_n {
                        for (j, &g) in gorder.iter().enumerate() {
                            sorted_ious[d * g_n + j] = p.ious[d * g_n + g];
                        }
                    }
                    let out: Vec<bool> = p.dt_area.iter().map(|&x| x < lo || x > hi).collect();
                    evals.push(match_images(
                        &sorted_ious,
                        &gi,
                        &cr,
                        &out,
                        1,
                        d_n,
                        g_n,
                        iou_thrs,
                    ));
                }
                if npig == 0 {
                    continue;
                }
                for (m, &max_det) in max_dets.iter().enumerate() {
                    // Every image's first max_det detections, concatenated in image order, then sorted by
                    // score (stable), as COCOeval.accumulate.
                    let mut rows: Vec<(f64, usize, usize)> = Vec::new();
                    for (pi, p) in pairs.iter().enumerate() {
                        for d in 0..p.dt_scores.len().min(max_det) {
                            rows.push((p.dt_scores[d], pi, d));
                        }
                    }
                    rows.sort_by(|x, y| {
                        (-x.0)
                            .partial_cmp(&-y.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let nd = rows.len();
                    for t in 0..t_n {
                        let (mut tp, mut fp) = (0u64, 0u64);
                        let mut rc = Vec::with_capacity(nd);
                        let mut pr = Vec::with_capacity(nd);
                        for &(_, pi, d) in &rows {
                            let d_n = pairs[pi].dt_scores.len();
                            let (dm, di) = (&evals[pi].0, &evals[pi].1);
                            let (matched, ignored) = (dm[t * d_n + d], di[t * d_n + d]);
                            if matched && !ignored {
                                tp += 1;
                            }
                            if !matched && !ignored {
                                fp += 1;
                            }
                            let (tpf, fpf) = (tp as f64, fp as f64);
                            rc.push(tpf / npig as f64);
                            pr.push(tpf / (fpf + tpf + f64::EPSILON));
                        }
                        rec[(t * a_n + a) * m_n + m] = rc.last().copied().unwrap_or(0.0);
                        for i in (0..nd.saturating_sub(1)).rev() {
                            if pr[i + 1] > pr[i] {
                                pr[i] = pr[i + 1];
                            }
                        }
                        for (r, &thr) in rec_thrs.iter().enumerate() {
                            let idx = rc.partition_point(|&x| x < thr);
                            prec[((t * r_n + r) * a_n + a) * m_n + m] =
                                if idx < nd { pr[idx] } else { 0.0 };
                        }
                    }
                }
            }
            (prec, rec)
        })
        .collect();
    let mut precision = vec![-1.0f64; t_n * r_n * k_n * a_n * m_n];
    let mut recall = vec![-1.0f64; t_n * k_n * a_n * m_n];
    for (k, (p, r)) in per_cat.iter().enumerate() {
        for t in 0..t_n {
            for a in 0..a_n {
                for m in 0..m_n {
                    recall[((t * k_n + k) * a_n + a) * m_n + m] = r[(t * a_n + a) * m_n + m];
                    for rr in 0..r_n {
                        precision[(((t * r_n + rr) * k_n + k) * a_n + a) * m_n + m] =
                            p[((t * r_n + rr) * a_n + a) * m_n + m];
                    }
                }
            }
        }
    }
    Evaluation {
        cats,
        precision,
        recall,
    }
}
