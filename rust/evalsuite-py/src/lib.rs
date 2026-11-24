//! `evalsuite._core`: Python bindings for evalsuite-core.
//!
//! Built against the stable ABI (abi3, Python >= 3.9), so one wheel per platform serves every supported Python.
//! Arrays are read in place through `(address, length)` views of checked, C-contiguous NumPy arrays (the buffer
//! protocol joined the stable ABI only in Python 3.11); token sequences cross as lists of integer ids. The Python package decides when to call these and keeps
//! its own implementation as the reference and fallback.

use evalsuite_core::{coco, meteor, resample, ter, text};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

fn chars(s: &str) -> Vec<u32> {
    s.chars().map(u32::from).collect()
}

/// BLEU statistics `[hyp_len, ref_len, correct_1..N, total_1..N]` for each segment.
#[pyfunction]
fn bleu_stats(
    py: Python<'_>,
    hyps: Vec<Vec<u32>>,
    refs: Vec<Vec<Vec<u32>>>,
    order: usize,
) -> PyResult<Vec<Vec<i64>>> {
    if hyps.len() != refs.len() {
        return Err(PyValueError::new_err(
            "hyps and refs must have the same length",
        ));
    }
    Ok(py.detach(|| {
        hyps.iter()
            .zip(refs.iter())
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
    if hyps.len() != refs.len() {
        return Err(PyValueError::new_err(
            "hyps and refs must have the same length",
        ));
    }
    Ok(py.detach(|| {
        let (mut edits, mut ref_len) = (0.0f64, 0.0f64);
        for (h, rs) in hyps.iter().zip(refs.iter()) {
            let best = rs.iter().map(|r| ter::ter_edits(h, r)).min().unwrap_or(0);
            edits += best as f64;
            let total: usize = rs.iter().map(|r| r.len()).sum();
            ref_len += total as f64 / rs.len().max(1) as f64;
        }
        (edits, ref_len)
    }))
}

/// Clipped n-gram overlaps for each (reference, prediction) pair.
#[pyfunction]
fn ngram_overlaps(py: Python<'_>, ts: Vec<Vec<u32>>, ps: Vec<Vec<u32>>, n: usize) -> Vec<i64> {
    py.detach(|| {
        ts.iter()
            .zip(ps.iter())
            .map(|(t, p)| text::ngram_overlap(t, p, n))
            .collect()
    })
}

/// Longest-common-subsequence lengths for each (reference, prediction) pair.
#[pyfunction]
fn lcs_lengths(py: Python<'_>, ts: Vec<Vec<u32>>, ps: Vec<Vec<u32>>) -> Vec<i64> {
    py.detach(|| {
        ts.iter()
            .zip(ps.iter())
            .map(|(t, p)| text::lcs_length(t, p) as i64)
            .collect()
    })
}

/// ROUGE-Lsum `(hits, reference tokens, prediction tokens)` for each (reference, prediction) pair of sentence lists.
#[pyfunction]
fn rouge_lsum_hits(
    py: Python<'_>,
    ts: Vec<Vec<Vec<u32>>>,
    ps: Vec<Vec<Vec<u32>>>,
) -> Vec<(i64, i64, i64)> {
    py.detach(|| {
        ts.iter()
            .zip(ps.iter())
            .map(|(t, p)| {
                let tt: Vec<&[u32]> = t.iter().map(|s| s.as_slice()).collect();
                let pp: Vec<&[u32]> = p.iter().map(|s| s.as_slice()).collect();
                text::rouge_lsum_hits(&tt, &pp)
            })
            .collect()
    })
}

/// Corpus chrF statistics (summed best-reference statistics of every example).
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
        let order = char_order + word_order;
        let mut total = vec![0.0f64; 3 * order];
        for i in 0..n {
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
            if let Some(best) =
                text::chrf_segment(&h, &rs, char_order, word_order, beta, eps_smoothing)
            {
                for (t, v) in total.iter_mut().zip(best) {
                    *t += v;
                }
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
    if hyps.len() != refs.len() {
        return Err(PyValueError::new_err(
            "hyps and refs must have the same length",
        ));
    }
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
        hyps.iter()
            .zip(refs.iter())
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
    // Booleans are read as bytes (any non-zero byte is true), never reinterpreted as Rust `bool`.
    let as_bools = |v: (usize, usize)| -> PyResult<Vec<bool>> {
        Ok(view::<u8>(v.0, v.1)?.iter().map(|&b| b != 0).collect())
    };
    let (gi, cr, out) = (as_bools(gign)?, as_bools(crowd)?, as_bools(dt_out)?);
    let (dtm, dt_ig) =
        py.detach(|| coco::match_images(ious, &gi, &cr, &out, n_images, d_max, g_max, &thresholds));
    let to_bytes = |v: Vec<bool>| -> Vec<u8> { v.into_iter().map(u8::from).collect() };
    Ok((
        PyBytes::new(py, &to_bytes(dtm)),
        PyBytes::new(py, &to_bytes(dt_ig)),
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(code_counts, m)?)?;
    Ok(())
}
