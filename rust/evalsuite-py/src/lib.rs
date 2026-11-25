//! `evalsuite._core`: Python bindings for evalsuite-core.
//!
//! Built against the stable ABI (abi3, Python >= 3.9), so one wheel per platform serves every supported Python.
//! Arrays are read in place through `(address, length)` views of checked, C-contiguous NumPy arrays (the buffer
//! protocol joined the stable ABI only in Python 3.11); token sequences cross as lists of integer ids. The
//! Python package decides when to call these and keeps its own implementation as the reference and fallback.
//!
//! Work runs on all CPU cores (rayon; `EVALSUITE_NUM_THREADS` limits it) with the GIL released. Parallel
//! loops only compute per-example results; every sum is then taken sequentially in example order, so results
//! are bit-identical for any number of threads.

use evalsuite_core::{
    agreement, coco, curve, groups, meteor, permutation, qa, ranks, resample, surface, ter, text,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// Smallest number of examples per parallel task: tiny corpora run on one thread without scheduling overhead.
const MIN_PER_TASK: usize = 16;

/// A read-only view of a C-contiguous NumPy array, passed as `(data address, element count)`.
///
/// The stable ABI on Python 3.9 has no buffer protocol, so the Python side (`evalsuite._native`) passes the
/// address of an array it has checked to be C-contiguous with exactly this element type, and keeps the array
/// referenced for the duration of the call. Only reads happen here.
fn view<'a, T>(addr: usize, len: usize) -> PyResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if addr == 0 || addr % std::mem::align_of::<T>() != 0 {
        return Err(PyValueError::new_err(
            "array data must be non-null and aligned",
        ));
    }
    // SAFETY: the caller guarantees `addr` points to `len` initialised, aligned, C-contiguous elements of `T`
    // that stay alive and are not written during this call (see the function comment above).
    Ok(unsafe { std::slice::from_raw_parts(addr as *const T, len) })
}

/// Booleans are read as bytes (any non-zero byte is true), never reinterpreted as Rust `bool`.
fn bools(v: (usize, usize)) -> PyResult<Vec<bool>> {
    Ok(view::<u8>(v.0, v.1)?.iter().map(|&b| b != 0).collect())
}

fn offsets(v: (usize, usize)) -> PyResult<Vec<usize>> {
    view::<i64>(v.0, v.1)?
        .iter()
        .map(|&x| {
            usize::try_from(x).map_err(|_| PyValueError::new_err("offsets must be non-negative"))
        })
        .collect()
}

fn chars(s: &str) -> Vec<u32> {
    s.chars().map(u32::from).collect()
}

fn same_len(a: usize, b: usize, what: &str) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "{what} must have the same length"
        )));
    }
    Ok(())
}

fn f64_bytes<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyBytes> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_ne_bytes());
    }
    PyBytes::new(py, &out)
}

/// BLEU statistics `[hyp_len, ref_len, correct_1..N, total_1..N]` for each segment.
#[pyfunction]
fn bleu_stats(
    py: Python<'_>,
    hyps: Vec<Vec<u32>>,
    refs: Vec<Vec<Vec<u32>>>,
    order: usize,
) -> PyResult<Vec<Vec<i64>>> {
    same_len(hyps.len(), refs.len(), "hyps and refs")?;
    Ok(py.detach(|| {
        hyps.par_iter()
            .zip(refs.par_iter())
            .with_min_len(MIN_PER_TASK)
            .map(|(h, rs)| {
                let rr: Vec<&[u32]> = rs.iter().map(|r| r.as_slice()).collect();
                text::bleu_stats(h, &rr, order)
            })
            .collect()
    }))
}

/// Self-BLEU statistics: each segment against all the others.
#[pyfunction]
fn self_bleu_stats(py: Python<'_>, toks: Vec<Vec<u32>>, order: usize) -> Vec<Vec<i64>> {
    py.detach(|| {
        (0..toks.len())
            .into_par_iter()
            .with_min_len(MIN_PER_TASK)
            .map(|i| {
                let others: Vec<&[u32]> = toks
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, t)| t.as_slice())
                    .collect();
                text::bleu_stats(&toks[i], &others, order)
            })
            .collect()
    })
}

/// Corpus TER totals `(edits, average reference length)`, summed in example order.
#[pyfunction]
fn ter_corpus(
    py: Python<'_>,
    hyps: Vec<Vec<u32>>,
    refs: Vec<Vec<Vec<u32>>>,
) -> PyResult<(f64, f64)> {
    same_len(hyps.len(), refs.len(), "hyps and refs")?;
    Ok(py.detach(|| {
        let per: Vec<(i64, f64)> = hyps
            .par_iter()
            .zip(refs.par_iter())
            .with_min_len(1)
            .map(|(h, rs)| {
                let best = rs.iter().map(|r| ter::ter_edits(h, r)).min().unwrap_or(0);
                let total: usize = rs.iter().map(|r| r.len()).sum();
                (best, total as f64 / rs.len().max(1) as f64)
            })
            .collect();
        let (mut edits, mut ref_len) = (0.0f64, 0.0f64);
        for (e, r) in per {
            edits += e as f64;
            ref_len += r;
        }
        (edits, ref_len)
    }))
}

/// Clipped n-gram overlaps for each (reference, prediction) pair.
#[pyfunction]
fn ngram_overlaps(
    py: Python<'_>,
    ts: Vec<Vec<u32>>,
    ps: Vec<Vec<u32>>,
    n: usize,
) -> PyResult<Vec<i64>> {
    same_len(ts.len(), ps.len(), "ts and ps")?;
    Ok(py.detach(|| {
        ts.par_iter()
            .zip(ps.par_iter())
            .with_min_len(MIN_PER_TASK)
            .map(|(t, p)| text::ngram_overlap(t, p, n))
            .collect()
    }))
}

/// Longest-common-subsequence lengths for each (reference, prediction) pair.
#[pyfunction]
fn lcs_lengths(py: Python<'_>, ts: Vec<Vec<u32>>, ps: Vec<Vec<u32>>) -> PyResult<Vec<i64>> {
    same_len(ts.len(), ps.len(), "ts and ps")?;
    Ok(py.detach(|| {
        ts.par_iter()
            .zip(ps.par_iter())
            .with_min_len(MIN_PER_TASK)
            .map(|(t, p)| text::lcs_length(t, p) as i64)
            .collect()
    }))
}

/// ROUGE-Lsum `(hits, reference tokens, prediction tokens)` for each (reference, prediction) pair of sentence lists.
#[pyfunction]
fn rouge_lsum_hits(
    py: Python<'_>,
    ts: Vec<Vec<Vec<u32>>>,
    ps: Vec<Vec<Vec<u32>>>,
) -> PyResult<Vec<(i64, i64, i64)>> {
    same_len(ts.len(), ps.len(), "ts and ps")?;
    Ok(py.detach(|| {
        ts.par_iter()
            .zip(ps.par_iter())
            .with_min_len(MIN_PER_TASK)
            .map(|(t, p)| {
                let tt: Vec<&[u32]> = t.iter().map(|s| s.as_slice()).collect();
                let pp: Vec<&[u32]> = p.iter().map(|s| s.as_slice()).collect();
                text::rouge_lsum_hits(&tt, &pp)
            })
            .collect()
    }))
}

/// Corpus chrF statistics (summed best-reference statistics of every example, in example order).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn chrf_corpus(
    py: Python<'_>,
    hyp_lines: Vec<String>,
    hyp_words: Vec<Vec<u32>>,
    ref_lines: Vec<Vec<String>>,
    ref_words: Vec<Vec<Vec<u32>>>,
    char_order: usize,
    word_order: usize,
    beta: f64,
    eps_smoothing: bool,
) -> PyResult<Vec<f64>> {
    let n = hyp_lines.len();
    if hyp_words.len() != n || ref_lines.len() != n || ref_words.len() != n {
        return Err(PyValueError::new_err(
            "all inputs must describe the same examples",
        ));
    }
    Ok(py.detach(|| {
        let per: Vec<Option<Vec<f64>>> = (0..n)
            .into_par_iter()
            .with_min_len(MIN_PER_TASK)
            .map(|i| {
                let hc = chars(&hyp_lines[i]);
                let h = text::ChrfText {
                    chars: &hc,
                    words: &hyp_words[i],
                };
                let rcs: Vec<Vec<u32>> = ref_lines[i].iter().map(|s| chars(s)).collect();
                let rs: Vec<text::ChrfText<'_>> = rcs
                    .iter()
                    .zip(ref_words[i].iter())
                    .map(|(c, w)| text::ChrfText { chars: c, words: w })
                    .collect();
                text::chrf_segment(&h, &rs, char_order, word_order, beta, eps_smoothing)
            })
            .collect();
        let mut total = vec![0.0f64; 3 * (char_order + word_order)];
        for best in per.into_iter().flatten() {
            for (t, v) in total.iter_mut().zip(best) {
                *t += v;
            }
        }
        total
    }))
}

/// chrF F-score from statistics (exposed for parity tests).
#[pyfunction]
fn chrf_f(stats: Vec<f64>, order: usize, beta: f64, eps_smoothing: bool) -> f64 {
    text::chrf_f(&stats, order, beta, eps_smoothing)
}

/// METEOR per example: the best score over its references. `stem_of[w]` is the stem id of word id `w`
/// (omit it to skip the stem stage).
#[pyfunction]
#[pyo3(signature = (hyps, refs, stem_of, alpha, beta, gamma))]
fn meteor_scores(
    py: Python<'_>,
    hyps: Vec<Vec<u32>>,
    refs: Vec<Vec<Vec<u32>>>,
    stem_of: Option<Vec<u32>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> PyResult<Vec<f64>> {
    same_len(hyps.len(), refs.len(), "hyps and refs")?;
    if let Some(st) = &stem_of {
        let max_id = hyps
            .iter()
            .flatten()
            .chain(refs.iter().flatten().flatten())
            .copied()
            .max();
        if max_id.is_some_and(|m| m as usize >= st.len()) {
            return Err(PyValueError::new_err("stem_of must cover every word id"));
        }
    }
    Ok(py.detach(|| {
        let stem = |ws: &[u32]| -> Vec<u32> {
            stem_of
                .as_ref()
                .map(|st| ws.iter().map(|&w| st[w as usize]).collect())
                .unwrap_or_default()
        };
        hyps.par_iter()
            .zip(refs.par_iter())
            .with_min_len(MIN_PER_TASK)
            .map(|(h, rs)| {
                let hs = stem(h);
                let mut best = f64::NEG_INFINITY;
                for r in rs {
                    let rst = stem(r);
                    let st = stem_of.as_ref().map(|_| (hs.as_slice(), rst.as_slice()));
                    let v = meteor::meteor_single(h, r, st, alpha, beta, gamma);
                    if v > best {
                        best = v;
                    }
                }
                best
            })
            .collect()
    }))
}

/// COCOeval greedy matching for padded images; returns `(dt_matched, dt_ignore)` as bytes of bools `[p][t][d]`.
/// Arrays are passed as `(address, length)` views: `ious` float64 `[p][d][g]`, `gign`/`crowd` bool `[p][g]`,
/// `dt_out` bool `[p][d]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn coco_match<'py>(
    py: Python<'py>,
    ious: (usize, usize),
    gign: (usize, usize),
    crowd: (usize, usize),
    dt_out: (usize, usize),
    n_images: usize,
    d_max: usize,
    g_max: usize,
    thresholds: Vec<f64>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    if ious.1 != n_images * d_max * g_max
        || gign.1 != n_images * g_max
        || crowd.1 != n_images * g_max
        || dt_out.1 != n_images * d_max
    {
        return Err(PyValueError::new_err(
            "array sizes do not match n_images, d_max and g_max",
        ));
    }
    let ious: &[f64] = view(ious.0, ious.1)?;
    let (gi, cr, out) = (bools(gign)?, bools(crowd)?, bools(dt_out)?);
    let (dtm, dt_ig) =
        py.detach(|| coco::match_images(ious, &gi, &cr, &out, n_images, d_max, g_max, &thresholds));
    let to_bytes = |v: Vec<bool>| -> Vec<u8> { v.into_iter().map(u8::from).collect() };
    Ok((
        PyBytes::new(py, &to_bytes(dtm)),
        PyBytes::new(py, &to_bytes(dt_ig)),
    ))
}

/// Full COCO bbox evaluation. Views: `gt_boxes`/`dt_boxes` float64 `[n][4]`, `gt_labels`/`dt_labels` int64,
/// `gt_crowd` bool, `gt_area`/`dt_scores` float64, `gt_off`/`dt_off` int64 image offsets (`n_images + 1`).
/// Returns `(classes, precision bytes [t][r][k][a][m], recall bytes [t][k][a][m])`.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn coco_evaluate<'py>(
    py: Python<'py>,
    gt_boxes: (usize, usize),
    gt_labels: (usize, usize),
    gt_crowd: (usize, usize),
    gt_area: (usize, usize),
    gt_off: (usize, usize),
    dt_boxes: (usize, usize),
    dt_labels: (usize, usize),
    dt_scores: (usize, usize),
    dt_off: (usize, usize),
    iou_thrs: Vec<f64>,
    max_dets: Vec<usize>,
    areas: Vec<(f64, f64)>,
    rec_thrs: Vec<f64>,
) -> PyResult<(Vec<i64>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let (go, doff) = (offsets(gt_off)?, offsets(dt_off)?);
    let (n_gt, n_dt) = (gt_labels.1, dt_labels.1);
    let monotone = |o: &[usize], n: usize| {
        !o.is_empty() && o[0] == 0 && *o.last().unwrap() == n && o.windows(2).all(|w| w[0] <= w[1])
    };
    if go.len() != doff.len()
        || !monotone(&go, n_gt)
        || !monotone(&doff, n_dt)
        || gt_boxes.1 != 4 * n_gt
        || gt_crowd.1 != n_gt
        || gt_area.1 != n_gt
        || dt_boxes.1 != 4 * n_dt
        || dt_scores.1 != n_dt
    {
        return Err(PyValueError::new_err("inconsistent detection arrays"));
    }
    let crowd = bools(gt_crowd)?;
    let img = coco::Images {
        gt_boxes: view(gt_boxes.0, gt_boxes.1)?,
        gt_labels: view(gt_labels.0, gt_labels.1)?,
        gt_crowd: &crowd,
        gt_area: view(gt_area.0, gt_area.1)?,
        gt_off: &go,
        dt_boxes: view(dt_boxes.0, dt_boxes.1)?,
        dt_labels: view(dt_labels.0, dt_labels.1)?,
        dt_scores: view(dt_scores.0, dt_scores.1)?,
        dt_off: &doff,
    };
    let ev = py.detach(|| coco::evaluate(&img, &iou_thrs, &max_dets, &areas, &rec_thrs));
    Ok((
        ev.cats,
        f64_bytes(py, &ev.precision),
        f64_bytes(py, &ev.recall),
    ))
}

/// Counts of each code `0..k` in every resample. `codes` is a uint32 view, `indices` an int64 view of
/// `rows × n` indices. Returns uint64 bytes (`rows × k`).
#[pyfunction]
fn code_counts<'py>(
    py: Python<'py>,
    codes: (usize, usize),
    k: usize,
    indices: (usize, usize),
    rows: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let codes: &[u32] = view(codes.0, codes.1)?;
    let idx: &[i64] = view(indices.0, indices.1)?;
    let out = py
        .detach(|| resample::code_counts(codes, k, idx, rows))
        .ok_or_else(|| PyValueError::new_err("index or code out of range"))?;
    let mut bytes = Vec::with_capacity(out.len() * 8);
    for v in out {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    Ok(PyBytes::new(py, &bytes))
}

/// Fill `out` (a writable, C-contiguous int64 NumPy array of `rows × n` elements, given as an `(address, length)`
/// view that the caller keeps alive and does not touch during the call) with bootstrap indices from the core's
/// seeded generator. `strata` is `(members, offsets)`: an int64 view of each stratum's members, concatenated, and
/// the stratum boundaries.
#[pyfunction]
#[pyo3(signature = (out, n, seed, strata=None))]
fn bootstrap_indices(
    py: Python<'_>,
    out: (usize, usize),
    n: usize,
    seed: u64,
    strata: Option<((usize, usize), Vec<usize>)>,
) -> PyResult<()> {
    let groups = match &strata {
        None => None,
        Some((members, offsets)) => {
            let m: &[i64] = view(members.0, members.1)?;
            let ok = offsets.first() == Some(&0)
                && offsets.last() == Some(&n)
                && offsets.windows(2).all(|w| w[0] < w[1])
                && m.len() == n
                && m.iter().all(|&i| i >= 0 && (i as usize) < n);
            if !ok {
                return Err(PyValueError::new_err("invalid strata"));
            }
            Some((m, offsets.as_slice()))
        }
    };
    if n == 0 || out.1 % n != 0 {
        return Err(PyValueError::new_err("output size must be a multiple of n"));
    }
    if out.1 > 0 && (out.0 == 0 || out.0 % std::mem::align_of::<i64>() != 0) {
        return Err(PyValueError::new_err("invalid output buffer"));
    }
    let dst: &mut [i64] = if out.1 == 0 {
        &mut []
    } else {
        // SAFETY: the caller passes a live, aligned, writable int64 buffer of exactly `out.1` elements that
        // nothing else reads or writes until this call returns (see `_native.array_view` and `resample_indices`).
        unsafe { std::slice::from_raw_parts_mut(out.0 as *mut i64, out.1) }
    };
    py.detach(|| resample::fill_bootstrap_indices(dst, n, seed, groups));
    Ok(())
}

/// An `(address, element count)` array view.
type View = (usize, usize);

/// ROC AUC and average precision of binary curves, one per column, in parallel. Each column is
/// `(y, score, weight)` float64 views of equal length. Returns `(auc, ap, positive weight, negative weight)` per
/// column; AUC/AP are NaN where undefined (the caller raises).
#[pyfunction]
fn curve_scores(
    py: Python<'_>,
    columns: Vec<(View, View, View)>,
) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    let mut cols: Vec<(&[f64], &[f64], &[f64])> = Vec::with_capacity(columns.len());
    for (y, s, w) in &columns {
        let (y, s, w): (&[f64], &[f64], &[f64]) =
            (view(y.0, y.1)?, view(s.0, s.1)?, view(w.0, w.1)?);
        same_len(y.len(), s.len(), "y and score")?;
        same_len(w.len(), s.len(), "weights and score")?;
        cols.push((y, s, w));
    }
    Ok(py.detach(|| {
        cols.par_iter()
            .map(|&(y, s, w)| curve::curve_scores(y, s, w))
            .collect()
    }))
}

/// Binary-curve scores `(auc, ap, positive weight, negative weight)` of every bootstrap resample: `y`, `score`
/// and optional `w` are float64 views of the full data, `indices` an int64 view of `rows × n` resample indices.
#[pyfunction]
#[pyo3(signature = (y, score, w, indices, rows))]
fn bootstrap_curves(
    py: Python<'_>,
    y: View,
    score: View,
    w: Option<View>,
    indices: View,
    rows: usize,
) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    let (y, s): (&[f64], &[f64]) = (view(y.0, y.1)?, view(score.0, score.1)?);
    same_len(y.len(), s.len(), "y and score")?;
    let w: Option<&[f64]> = match w {
        Some(v) => {
            let w: &[f64] = view(v.0, v.1)?;
            same_len(w.len(), s.len(), "weights and score")?;
            Some(w)
        }
        None => None,
    };
    let idx: &[i64] = view(indices.0, indices.1)?;
    if rows == 0 || idx.len() % rows != 0 {
        return Err(PyValueError::new_err("indices must hold rows × n entries"));
    }
    let len = s.len() as i64;
    if idx.iter().any(|&i| i < 0 || i >= len) {
        return Err(PyValueError::new_err("resample index out of range"));
    }
    Ok(py.detach(|| curve::bootstrap_curves(y, s, w, idx, rows)))
}

/// Leave-one-block-out binary-curve scores (jackknife): `group_of` is a uint32 view giving each observation's
/// block in `0..groups`.
#[pyfunction]
#[pyo3(signature = (y, score, w, group_of, groups))]
fn leave_group_out_curves(
    py: Python<'_>,
    y: View,
    score: View,
    w: Option<View>,
    group_of: View,
    groups: usize,
) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    let (y, s): (&[f64], &[f64]) = (view(y.0, y.1)?, view(score.0, score.1)?);
    same_len(y.len(), s.len(), "y and score")?;
    let w: Option<&[f64]> = match w {
        Some(v) => {
            let w: &[f64] = view(v.0, v.1)?;
            same_len(w.len(), s.len(), "weights and score")?;
            Some(w)
        }
        None => None,
    };
    let g: &[u32] = view(group_of.0, group_of.1)?;
    same_len(g.len(), s.len(), "group_of and score")?;
    if g.iter().any(|&x| x as usize >= groups) {
        return Err(PyValueError::new_err("group index out of range"));
    }
    Ok(py.detach(|| curve::leave_group_out_curves(y, s, w, g, groups)))
}

/// Four float64 byte buffers: tz of positives, tz of negatives, tx, ty.
type RankBytes<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);

/// DeLong rankings of each score column (float64 views) against binary labels (a bool view): per column
/// `(tz positives, tz negatives, tx, ty)` as float64 bytes, columns in parallel.
#[pyfunction]
fn delong_ranks<'py>(
    py: Python<'py>,
    positive: View,
    columns: Vec<View>,
) -> PyResult<Vec<RankBytes<'py>>> {
    let p: Vec<bool> = bools(positive)?;
    let mut cols: Vec<&[f64]> = Vec::with_capacity(columns.len());
    for c in &columns {
        let s: &[f64] = view(c.0, c.1)?;
        same_len(s.len(), p.len(), "scores and labels")?;
        cols.push(s);
    }
    let out: Vec<_> = py.detach(|| {
        cols.par_iter()
            .map(|s| ranks::delong_ranks(&p, s))
            .collect()
    });
    Ok(out
        .into_iter()
        .map(|(a, b, c, d)| {
            (
                f64_bytes(py, &a),
                f64_bytes(py, &b),
                f64_bytes(py, &c),
                f64_bytes(py, &d),
            )
        })
        .collect())
}

/// Exact-match (`f1 = false`) or token-F1 score per prediction against its references (best reference),
/// with SQuAD normalisation when `normalize`. `None` when any text is non-ASCII (the caller uses Python).
#[pyfunction]
fn qa_scores(
    py: Python<'_>,
    preds: Vec<String>,
    refs: Vec<Vec<String>>,
    normalize: bool,
    f1: bool,
) -> PyResult<Option<Vec<f64>>> {
    same_len(preds.len(), refs.len(), "predictions and references")?;
    Ok(py.detach(|| {
        let prep = |t: &str| -> Option<String> {
            if normalize {
                qa::normalize(t)
            } else if t.is_ascii() {
                Some(t.to_owned())
            } else {
                None
            }
        };
        let p: Option<Vec<String>> = preds.iter().map(|t| prep(t)).collect();
        let r: Option<Vec<Vec<String>>> = refs
            .iter()
            .map(|rs| rs.iter().map(|t| prep(t)).collect())
            .collect();
        let (p, r) = (p?, r?);
        Some(
            p.par_iter()
                .zip(r.par_iter())
                .with_min_len(256)
                .map(|(h, rs)| {
                    if f1 {
                        rs.iter()
                            .map(|x| qa::f1(h, x))
                            .fold(f64::NEG_INFINITY, f64::max)
                    } else {
                        f64::from(u8::from(rs.iter().any(|x| x == h)))
                    }
                })
                .collect(),
        )
    }))
}

/// SQuAD normalisation of each text; `None` for non-ASCII texts (normalise those in Python).
#[pyfunction]
fn squad_normalize(py: Python<'_>, texts: Vec<String>) -> Vec<Option<String>> {
    py.detach(|| texts.iter().map(|t| qa::normalize(t)).collect())
}

/// Nearest-surface distances both ways (A→B, B→A, float64 bytes, in `np.argwhere` order of each surface)
/// between two non-empty binary masks (bool views) of shape `dims` (2 or 3 axes), with `spacing` per axis.
#[pyfunction]
fn surface_distances<'py>(
    py: Python<'py>,
    a: View,
    b: View,
    dims: Vec<usize>,
    spacing: Vec<f64>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    if !(2..=3).contains(&dims.len()) || spacing.len() != dims.len() {
        return Err(PyValueError::new_err(
            "masks must have 2 or 3 axes, with one spacing per axis",
        ));
    }
    if spacing.iter().any(|&s| !(s > 0.0 && s.is_finite())) {
        return Err(PyValueError::new_err("spacing must be positive and finite"));
    }
    let size = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let (ma, mb) = (bools(a)?, bools(b)?);
    if size != Some(ma.len()) || ma.len() != mb.len() {
        return Err(PyValueError::new_err("mask size does not match dims"));
    }
    let (da, db) = py.detach(|| {
        let sa = surface::surface_points(&ma, &dims);
        let sb = surface::surface_points(&mb, &dims);
        if sa.is_empty() || sb.is_empty() {
            return (Vec::new(), Vec::new());
        }
        (
            surface::nearest(&sa, &sb, &dims, &spacing),
            surface::nearest(&sb, &sa, &dims, &spacing),
        )
    });
    Ok((f64_bytes(py, &da), f64_bytes(py, &db)))
}

/// Calibration bin sums `(Σw, Σw·y, Σw·p)` per bin (`len(inner_edges) + 2` bins, as `np.bincount` with
/// `minlength = len(edges)`), from float64 views of `y`, `p` and `w` and the inner bin edges.
#[pyfunction]
fn calibration_bins(
    py: Python<'_>,
    y: View,
    p: View,
    w: View,
    inner_edges: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (y, p, w): (&[f64], &[f64], &[f64]) = (view(y.0, y.1)?, view(p.0, p.1)?, view(w.0, w.1)?);
    same_len(y.len(), p.len(), "y and p")?;
    same_len(w.len(), p.len(), "weights and p")?;
    let [a, b, c] = py.detach(|| groups::calibration_bins(y, p, w, &inner_edges));
    Ok((a, b, c))
}

/// Permutation-test count for the split statistic `2·Σ_group s − total >= threshold` over groups of size `k`:
/// every combination when `trials` is `None` (exact), else `trials` seeded random groups.
#[pyfunction]
#[pyo3(signature = (s, k, total, threshold, trials=None, seed=0))]
fn split_permutation_count(
    py: Python<'_>,
    s: Vec<f64>,
    k: usize,
    total: f64,
    threshold: f64,
    trials: Option<usize>,
    seed: u64,
) -> PyResult<u64> {
    if k > s.len() {
        return Err(PyValueError::new_err("group larger than the items"));
    }
    Ok(py.detach(|| match trials {
        None => permutation::exact_count(&s, k, total, threshold),
        Some(t) => permutation::random_count(&s, k, total, threshold, t, seed),
    }))
}

/// Agreement within each group of answer ids: `(unanimous, equal pairs, pairs, modal count, mode)` per group.
#[pyfunction]
fn group_stats(py: Python<'_>, groups: Vec<Vec<u32>>) -> Vec<(bool, u64, u64, u64, u32)> {
    py.detach(|| {
        groups
            .par_iter()
            .with_min_len(64)
            .map(|g| {
                let s = groups::group_stats(g);
                (s.unanimous, s.equal_pairs, s.pairs, s.modal_count, s.mode)
            })
            .collect()
    })
}

/// Per row of numbers (read directly from the Python sequences), how many values lie strictly below the row's
/// threshold; `None` if any value or threshold is NaN or infinite.
#[pyfunction]
fn count_below(
    py: Python<'_>,
    rows: Vec<Vec<f64>>,
    thresholds: Vec<f64>,
) -> PyResult<Option<Vec<u64>>> {
    same_len(rows.len(), thresholds.len(), "rows and thresholds")?;
    Ok(py.detach(|| groups::count_below(&rows, &thresholds)))
}

/// Per-row statistics of ragged rows of numbers (each row a sequence of floats, read directly from the Python
/// objects): `(NumPy sum, min, max, count > 0)` per row, or `None` if a value is NaN or infinite. Rows that are
/// not sequences of numbers raise `TypeError` (the caller falls back to its Python path).
#[pyfunction]
fn ragged_stats(py: Python<'_>, rows: Vec<Vec<f64>>) -> Option<Vec<(f64, f64, f64, u64)>> {
    py.detach(|| {
        if rows.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        Some(
            rows.par_iter()
                .with_min_len(256)
                .map(|r| groups::row_stats(r))
                .collect(),
        )
    })
}

fn check_players(rows: &[(usize, usize, f64)], m: usize) -> PyResult<()> {
    if rows.iter().any(|&(a, b, _)| a >= m || b >= m) {
        return Err(PyValueError::new_err("player index out of range"));
    }
    Ok(())
}

/// Sequential Elo ratings of `m` players from `(a, b, outcome for a)` rows.
#[pyfunction]
fn elo_ratings(
    rows: Vec<(usize, usize, f64)>,
    m: usize,
    k: f64,
    initial: f64,
    scale: f64,
    base: f64,
) -> PyResult<Vec<f64>> {
    check_players(&rows, m)?;
    Ok(agreement::elo(&rows, m, k, initial, scale, base))
}

/// Bradley–Terry win matrix (m × m, row-major) from `(a, b, outcome for a)` rows.
#[pyfunction]
fn win_matrix(rows: Vec<(usize, usize, f64)>, m: usize) -> PyResult<Vec<f64>> {
    check_players(&rows, m)?;
    Ok(agreement::win_matrix(&rows, m))
}

/// `compensated` (below) must be `sys.version_info >= (3, 12)`: it selects the float summation of Python's
/// built-in `sum()` so results match the running interpreter (see `evalsuite_core::text::PySum`).
///
/// CIDEr-D per-order totals for each prediction (see `evalsuite_core::text::cider_totals`).
#[pyfunction]
fn cider_totals(
    py: Python<'_>,
    hyps: Vec<Vec<u32>>,
    refs: Vec<Vec<Vec<u32>>>,
    n: usize,
    sigma: f64,
    compensated: bool,
) -> PyResult<Vec<Vec<f64>>> {
    same_len(hyps.len(), refs.len(), "hyps and refs")?;
    Ok(py.detach(|| {
        let h: Vec<&[u32]> = hyps.iter().map(|x| x.as_slice()).collect();
        let r: Vec<Vec<&[u32]>> = refs
            .iter()
            .map(|rs| rs.iter().map(|x| x.as_slice()).collect())
            .collect();
        text::cider_totals(&h, &r, n, sigma, compensated)
    }))
}

/// CodeBLEU's corpus BLEU statistics `(num[4], den[4], hyp_len, ref_len)`, accumulated in example order.
/// `keyword[w]` marks keyword token ids for the keyword-weighted variant (omit for plain BLEU).
#[pyfunction]
#[pyo3(signature = (hyps, refs, keyword, compensated))]
#[allow(clippy::type_complexity)]
fn codebleu_stats(
    py: Python<'_>,
    hyps: Vec<Vec<u32>>,
    refs: Vec<Vec<Vec<u32>>>,
    keyword: Option<Vec<bool>>,
    compensated: bool,
) -> PyResult<(Vec<f64>, Vec<f64>, i64, i64)> {
    same_len(hyps.len(), refs.len(), "hyps and refs")?;
    if let Some(kw) = &keyword {
        let max_id = refs.iter().flatten().flatten().copied().max();
        if max_id.is_some_and(|m| m as usize >= kw.len()) {
            return Err(PyValueError::new_err("keyword must cover every token id"));
        }
    }
    Ok(py.detach(|| {
        let per: Vec<_> = hyps
            .par_iter()
            .zip(refs.par_iter())
            .with_min_len(MIN_PER_TASK)
            .map(|(h, rs)| {
                let rr: Vec<&[u32]> = rs.iter().map(|r| r.as_slice()).collect();
                text::codebleu_example(h, &rr, keyword.as_deref(), compensated)
            })
            .collect();
        let (mut num, mut den) = (vec![0.0f64; 4], vec![0.0f64; 4]);
        let (mut hl, mut rl) = (0i64, 0i64);
        for (terms, h, r) in per {
            for (n, ts) in terms.iter().enumerate() {
                for &(a, b) in ts {
                    num[n] += a;
                    den[n] += b;
                }
            }
            hl += h;
            rl += r;
        }
        (num, den, hl, rl)
    }))
}

/// `(distinct n-grams, total n-grams)` over all sequences.
#[pyfunction]
fn distinct_ngrams(py: Python<'_>, seqs: Vec<Vec<u32>>, n: usize) -> (u64, u64) {
    py.detach(|| {
        let s: Vec<&[u32]> = seqs.iter().map(|x| x.as_slice()).collect();
        text::distinct_ngrams(&s, n)
    })
}

/// Number of threads the core uses.
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// Ranked-retrieval scores per query (precision / recall / hit rate @k, MRR, AP, NDCG) straight from the
/// Python objects: the judgements and rankings are validated and looked up with Python's own `dict` / `set`
/// semantics (hash and `==`), without the interpreter loop. Returns `(validation error, per-query scores,
/// queries without relevant documents)`; arithmetic follows the Python code operation by operation (Python's
/// `sum()` included), so the scores are bit-identical.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn retrieval_scores(
    py: Python<'_>,
    kind: &str,
    relevant: Vec<Bound<'_, PyAny>>,
    retrieved: Vec<Bound<'_, PyAny>>,
    k: Option<usize>,
    exponential: bool,
    compensated: bool,
) -> PyResult<(Option<String>, Vec<f64>, Vec<usize>)> {
    use pyo3::types::{PyBytes as Bytes, PyDict, PyFloat, PySet, PyString};
    if relevant.len() != retrieved.len() {
        return Ok((
            Some(format!(
                "relevant and retrieved must have one entry per query. Received {} and {}.",
                relevant.len(),
                retrieved.len()
            )),
            vec![],
            vec![],
        ));
    }
    let mapping = py.import("collections.abc")?.getattr("Mapping")?;
    let generic = py.import("numpy")?.getattr("generic")?;
    let float = py.import("builtins")?.getattr("float")?;
    let mut queries: Vec<(Bound<'_, PyDict>, Vec<Bound<'_, PyAny>>)> =
        Vec::with_capacity(relevant.len());
    for (i, (r, d)) in relevant.iter().zip(&retrieved).enumerate() {
        let judged = PyDict::new(py);
        if r.is_instance(&mapping)? {
            for item in r.call_method0("items")?.try_iter()? {
                let (key, v): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
                let g: f64 = float.call1((&v,))?.cast_into::<PyFloat>()?.value();
                if !g.is_finite() || g < 0.0 {
                    return Ok((
                        Some(format!(
                            "relevant[{i}]: relevance grades must be finite and >= 0; got {} for {}.",
                            v.repr()?,
                            key.repr()?
                        )),
                        vec![],
                        vec![],
                    ));
                }
                if g > 0.0 {
                    judged.set_item(key, g)?;
                }
            }
        } else if r.is_instance_of::<PyString>() || r.is_instance_of::<Bytes>() {
            return Ok((
                Some(format!(
                    "relevant[{i}] must be a set, list or mapping of relevant ids, not a string."
                )),
                vec![],
                vec![],
            ));
        } else {
            for key in r.try_iter()? {
                judged.set_item(key?, 1.0)?;
            }
        }
        if d.is_instance_of::<PyString>()
            || d.is_instance_of::<Bytes>()
            || !d.hasattr("__iter__")?
        {
            return Ok((
                Some(format!("retrieved[{i}] must be a ranked list of ids.")),
                vec![],
                vec![],
            ));
        }
        let ranking: Vec<Bound<'_, PyAny>> = d.try_iter()?.collect::<PyResult<_>>()?;
        let seen = PySet::empty(py)?;
        for x in &ranking {
            if x.is_instance(&generic)? {
                seen.add(x.call_method0("item")?)?;
            } else {
                seen.add(x)?;
            }
        }
        if seen.len() != ranking.len() {
            return Ok((
                Some(format!("retrieved[{i}] lists the same id more than once.")),
                vec![],
                vec![],
            ));
        }
        queries.push((judged, ranking));
    }
    let empty: Vec<usize> = queries
        .iter()
        .enumerate()
        .filter(|(_, q)| q.0.is_empty())
        .map(|(i, _)| i)
        .collect();
    let needs_relevant = matches!(kind, "recall" | "map" | "ndcg");
    if needs_relevant && !empty.is_empty() {
        return Ok((None, vec![], empty));
    }
    let mut scores = Vec::with_capacity(queries.len());
    for (rel, run) in &queries {
        let top = match k {
            Some(k) => &run[..k.min(run.len())],
            None => &run[..],
        };
        let score = match kind {
            "precision" | "recall" | "hit" => {
                let mut hits = 0usize;
                for x in top {
                    hits += usize::from(rel.contains(x)?);
                }
                match kind {
                    "precision" => hits as f64 / k.unwrap_or(1) as f64,
                    "recall" => hits as f64 / rel.len() as f64,
                    _ => f64::from(u8::from(hits > 0)),
                }
            }
            "mrr" => {
                let mut rr = 0.0;
                for (i, x) in top.iter().enumerate() {
                    if rel.contains(x)? {
                        rr = 1.0 / (i + 1) as f64;
                        break;
                    }
                }
                rr
            }
            "map" => {
                let (mut hits, mut total) = (0usize, 0.0f64);
                for (i, x) in top.iter().enumerate() {
                    if rel.contains(x)? {
                        hits += 1;
                        total += hits as f64 / (i + 1) as f64;
                    }
                }
                total / rel.len() as f64
            }
            "ndcg" => {
                let g = |x: f64| if exponential { 2.0f64.powf(x) - 1.0 } else { x };
                let mut dcg = text::PySum::new(compensated);
                for (i, x) in top.iter().enumerate() {
                    let grade = match rel.get_item(x)? {
                        Some(v) => v.extract::<f64>()?,
                        None => 0.0,
                    };
                    dcg.add(g(grade) / ((i + 2) as f64).log2());
                }
                let mut ideal: Vec<f64> = rel
                    .values()
                    .iter()
                    .map(|v| v.extract::<f64>())
                    .collect::<PyResult<_>>()?;
                // `sorted(..., reverse=True)` is stable; equal grades are interchangeable here.
                ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(k) = k {
                    ideal.truncate(k);
                }
                let mut idcg = text::PySum::new(compensated);
                for (i, x) in ideal.iter().enumerate() {
                    idcg.add(g(*x) / ((i + 2) as f64).log2());
                }
                dcg.value() / idcg.value()
            }
            _ => return Err(PyValueError::new_err("unknown retrieval metric")),
        };
        scores.push(score);
    }
    Ok((None, scores, empty))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // EVALSUITE_NUM_THREADS caps the worker threads (default: one per CPU core). Read once, at import.
    if let Some(n) = std::env::var("EVALSUITE_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        if n > 0 {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(bleu_stats, m)?)?;
    m.add_function(wrap_pyfunction!(self_bleu_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ter_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(rouge_lsum_hits, m)?)?;
    m.add_function(wrap_pyfunction!(chrf_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(chrf_f, m)?)?;
    m.add_function(wrap_pyfunction!(meteor_scores, m)?)?;
    m.add_function(wrap_pyfunction!(coco_match, m)?)?;
    m.add_function(wrap_pyfunction!(coco_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(code_counts, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(curve_scores, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_curves, m)?)?;
    m.add_function(wrap_pyfunction!(leave_group_out_curves, m)?)?;
    m.add_function(wrap_pyfunction!(delong_ranks, m)?)?;
    m.add_function(wrap_pyfunction!(qa_scores, m)?)?;
    m.add_function(wrap_pyfunction!(surface_distances, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_bins, m)?)?;
    m.add_function(wrap_pyfunction!(split_permutation_count, m)?)?;
    m.add_function(wrap_pyfunction!(squad_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(group_stats, m)?)?;
    m.add_function(wrap_pyfunction!(count_below, m)?)?;
    m.add_function(wrap_pyfunction!(elo_ratings, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_stats, m)?)?;
    m.add_function(wrap_pyfunction!(win_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(cider_totals, m)?)?;
    m.add_function(wrap_pyfunction!(codebleu_stats, m)?)?;
    m.add_function(wrap_pyfunction!(distinct_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(retrieval_scores, m)?)?;
    Ok(())
}
