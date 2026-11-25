//! Embedding-based metrics over embeddings you computed: BERTScore, sentence-embedding similarity,
//! MoverScore (exact earth mover's distance), MAUVE, and an adapter for learned metrics (COMET, BLEURT, …), as
//! EvalSuite's Python `evalsuite.text.semantic`.
//!
//! Matrices are row-major `Vec<Vec<f64>>`: one row per token (BERTScore, MoverScore) or per text.
//!
//! ```
//! use evalsuite::semantic::{bertscore, BertMeasure};
//! let r = vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]];
//! let s = bertscore(&r, &r, None, None, BertMeasure::F1, None).unwrap();
//! assert!((s[0] - 1.0).abs() < 1e-12);
//! ```

use crate::error::{invalid, Result};
use crate::util::np_sum;

type Matrix = Vec<Vec<f64>>;

fn matrices(x: &[Matrix], name: &str) -> Result<usize> {
    if x.is_empty() {
        return invalid(format!("{name} is empty."));
    }
    let dim = x[0].first().map_or(0, Vec::len);
    for (i, m) in x.iter().enumerate() {
        if m.is_empty() || m[0].is_empty() {
            return invalid(format!(
                "{name}[{i}] must be a non-empty 2-D array (tokens × dim)."
            ));
        }
        if m.iter().any(|r| r.len() != dim) {
            return invalid(format!(
                "{name}[{i}] has dimension {}; expected {dim} like the others.",
                m[0].len()
            ));
        }
        if m.iter().flatten().any(|v| !v.is_finite()) {
            return invalid(format!("{name}[{i}] contains NaN or infinite values."));
        }
    }
    Ok(dim)
}

fn vectors(x: &[Vec<f64>], name: &str) -> Result<usize> {
    let dim = x.first().map_or(0, Vec::len);
    if x.is_empty() || dim == 0 || x.iter().any(|r| r.len() != dim) {
        return invalid(format!(
            "{name} must be a 2-D array with one embedding per row."
        ));
    }
    if x.iter().flatten().any(|v| !v.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite values."));
    }
    Ok(dim)
}

fn weights(w: Option<&[f64]>, n: usize, name: &str) -> Result<Vec<f64>> {
    match w {
        None => Ok(vec![1.0; n]),
        Some(a) => {
            if a.len() != n {
                return invalid(format!(
                    "{name} must have one weight per token ({n}); got {}.",
                    a.len()
                ));
            }
            if a.iter().any(|&x| !x.is_finite() || x < 0.0) || np_sum(a) <= 0.0 {
                return invalid(format!(
                    "{name} must be finite, non-negative and not all zero."
                ));
            }
            Ok(a.to_vec())
        }
    }
}

fn unit(a: &[Vec<f64>]) -> Result<Matrix> {
    a.iter()
        .map(|r| {
            let n = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n == 0.0 {
                return invalid(
                    "Embeddings must not be all-zero vectors (cosine similarity is undefined).",
                );
            }
            Ok(r.iter().map(|x| x / n).collect())
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Which BERTScore component to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BertMeasure {
    #[default]
    F1,
    Precision,
    Recall,
}

/// Per-example BERTScore from token embeddings (greedy cosine matching), with optional IDF weights per token
/// and an optional baseline `b` for rescaling as `(s − b) / (1 − b)`.
pub fn bertscore(
    reference_embeddings: &[Matrix],
    prediction_embeddings: &[Matrix],
    reference_weights: Option<&[Vec<f64>]>,
    prediction_weights: Option<&[Vec<f64>]>,
    measure: BertMeasure,
    baseline: Option<f64>,
) -> Result<Vec<f64>> {
    let dr = matrices(reference_embeddings, "reference_embeddings")?;
    let dp = matrices(prediction_embeddings, "prediction_embeddings")?;
    if reference_embeddings.len() != prediction_embeddings.len() {
        return invalid(format!(
            "reference_embeddings and prediction_embeddings must contain the same number of examples. Received {} and {}.",
            reference_embeddings.len(),
            prediction_embeddings.len()
        ));
    }
    if dr != dp {
        return invalid("Reference and prediction embeddings must have the same dimension.");
    }
    let n = reference_embeddings.len();
    if reference_weights.is_some_and(|w| w.len() != n)
        || prediction_weights.is_some_and(|w| w.len() != n)
    {
        return invalid("Provide one weight array per example (or none).");
    }
    (0..n)
        .map(|k| {
            let (r, p) = (
                unit(&reference_embeddings[k])?,
                unit(&prediction_embeddings[k])?,
            );
            let wr = weights(
                reference_weights.map(|w| w[k].as_slice()),
                r.len(),
                "reference_weights",
            )?;
            let wp = weights(
                prediction_weights.map(|w| w[k].as_slice()),
                p.len(),
                "prediction_weights",
            )?;
            let sim: Vec<Vec<f64>> = r
                .iter()
                .map(|a| p.iter().map(|b| dot(a, b)).collect())
                .collect();
            let col_max: Vec<f64> = (0..p.len())
                .map(|j| {
                    sim.iter()
                        .map(|row| row[j])
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            let row_max: Vec<f64> = sim
                .iter()
                .map(|row| row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .collect();
            let mut precision = np_sum(
                &col_max
                    .iter()
                    .zip(&wp)
                    .map(|(a, b)| a * b)
                    .collect::<Vec<_>>(),
            ) / np_sum(&wp);
            let mut recall = np_sum(
                &row_max
                    .iter()
                    .zip(&wr)
                    .map(|(a, b)| a * b)
                    .collect::<Vec<_>>(),
            ) / np_sum(&wr);
            if let Some(b) = baseline {
                precision = (precision - b) / (1.0 - b);
                recall = (recall - b) / (1.0 - b);
            }
            Ok(match measure {
                BertMeasure::Precision => precision,
                BertMeasure::Recall => recall,
                BertMeasure::F1 => {
                    if precision + recall == 0.0 {
                        0.0
                    } else {
                        2.0 * precision * recall / (precision + recall)
                    }
                }
            })
        })
        .collect()
}

/// Similarity or distance for [`embedding_similarity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Similarity {
    #[default]
    Cosine,
    Euclidean,
    Manhattan,
}

/// Row-wise similarity of two `(n, dim)` embedding arrays; `normalize` scales rows to unit length before a
/// distance.
pub fn embedding_similarity(
    reference_embeddings: &[Vec<f64>],
    prediction_embeddings: &[Vec<f64>],
    metric: Similarity,
    normalize: bool,
) -> Result<Vec<f64>> {
    let da = vectors(reference_embeddings, "reference_embeddings")?;
    let db = vectors(prediction_embeddings, "prediction_embeddings")?;
    if da != db || reference_embeddings.len() != prediction_embeddings.len() {
        return invalid("Embedding arrays must have the same shape.");
    }
    let (a, b) = if metric == Similarity::Cosine || normalize {
        (unit(reference_embeddings)?, unit(prediction_embeddings)?)
    } else {
        (
            reference_embeddings.to_vec(),
            prediction_embeddings.to_vec(),
        )
    };
    Ok(a.iter()
        .zip(&b)
        .map(|(x, y)| match metric {
            Similarity::Cosine => dot(x, y),
            Similarity::Euclidean => x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt(),
            Similarity::Manhattan => x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum(),
        })
        .collect())
}

/// Exact earth mover's distance between histograms `a` (rows) and `b` (columns), both summing to 1, under a
/// dense cost matrix (successive shortest paths with Dijkstra and potentials; exact up to rounding).
pub fn emd(a: &[f64], b: &[f64], cost: &[Vec<f64>]) -> f64 {
    let (n, m) = (a.len(), b.len());
    let eps = 1e-15;
    let mut ra = a.to_vec();
    let mut rb = b.to_vec();
    let mut flow = vec![vec![0.0f64; m]; n];
    let (mut pr, mut pc) = (vec![0.0f64; n], vec![0.0f64; m]);
    // Potential of the super-source; its edges to rows with supply left have reduced cost ps − pr[i] ≥ 0.
    let ps = 0.0f64;
    loop {
        if ra.iter().all(|&x| x <= eps) || rb.iter().all(|&x| x <= eps) {
            break;
        }
        // Dijkstra over rows (0..n) and columns (n..n+m) with reduced costs.
        let inf = f64::INFINITY;
        let mut dist = vec![inf; n + m];
        let mut parent = vec![usize::MAX; n + m];
        let mut done = vec![false; n + m];
        for i in 0..n {
            if ra[i] > eps {
                dist[i] = (ps - pr[i]).max(0.0);
            }
        }
        loop {
            let mut u = usize::MAX;
            let mut best = inf;
            for v in 0..n + m {
                if !done[v] && dist[v] < best {
                    best = dist[v];
                    u = v;
                }
            }
            if u == usize::MAX {
                break;
            }
            done[u] = true;
            if u < n {
                for j in 0..m {
                    let d = best + cost[u][j] + pr[u] - pc[j];
                    if !done[n + j] && d < dist[n + j] {
                        dist[n + j] = d;
                        parent[n + j] = u;
                    }
                }
            } else {
                let j = u - n;
                for i in 0..n {
                    if !done[i] && flow[i][j] > eps {
                        let d = best - cost[i][j] + pc[j] - pr[i];
                        if d < dist[i] {
                            dist[i] = d;
                            parent[i] = u;
                        }
                    }
                }
            }
        }
        let target = (0..m)
            .filter(|&j| rb[j] > eps && dist[n + j] < inf)
            .min_by(|&x, &y| dist[n + x].partial_cmp(&dist[n + y]).unwrap());
        let Some(t) = target else { break };
        // Nodes not reached stay unreachable (new residual edges only appear along the path), so only the
        // reached ones need their potential updated.
        for i in 0..n {
            if dist[i].is_finite() {
                pr[i] += dist[i];
            }
        }
        for j in 0..m {
            if dist[n + j].is_finite() {
                pc[j] += dist[n + j];
            }
        }
        // Walk back to the source row, finding the bottleneck.
        let mut amount = rb[t];
        let mut v = n + t;
        while parent[v] != usize::MAX {
            let u = parent[v];
            if u >= n {
                amount = amount.min(flow[v][u - n]);
            }
            v = u;
        }
        amount = amount.min(ra[v]);
        let start = v;
        let mut v = n + t;
        while parent[v] != usize::MAX {
            let u = parent[v];
            if u < n {
                flow[u][v - n] += amount;
            } else {
                flow[v][u - n] -= amount;
            }
            v = u;
        }
        ra[start] -= amount;
        rb[t] -= amount;
    }
    let terms: Vec<f64> = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| flow[i][j] * cost[i][j])
        .collect();
    terms.iter().sum()
}

/// Per-example MoverScore: 1 − EMD between the (optionally IDF-weighted) token embeddings, with Euclidean
/// cost between L2-normalised embeddings.
pub fn moverscore(
    reference_embeddings: &[Matrix],
    prediction_embeddings: &[Matrix],
    reference_weights: Option<&[Vec<f64>]>,
    prediction_weights: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    matrices(reference_embeddings, "reference_embeddings")?;
    matrices(prediction_embeddings, "prediction_embeddings")?;
    if reference_embeddings.len() != prediction_embeddings.len() {
        return invalid(
            "reference_embeddings and prediction_embeddings must have the same length.",
        );
    }
    (0..reference_embeddings.len())
        .map(|k| {
            let (r, p) = (&reference_embeddings[k], &prediction_embeddings[k]);
            let wr = weights(
                reference_weights.map(|w| w[k].as_slice()),
                r.len(),
                "reference_weights",
            )?;
            let wp = weights(
                prediction_weights.map(|w| w[k].as_slice()),
                p.len(),
                "prediction_weights",
            )?;
            let (ru, pu) = (unit(r)?, unit(p)?);
            let cost: Vec<Vec<f64>> = ru
                .iter()
                .map(|x| {
                    pu.iter()
                        .map(|y| {
                            let c = -2.0 * dot(x, y) + dot(x, x) + dot(y, y);
                            c.max(0.0).sqrt()
                        })
                        .collect()
                })
                .collect();
            let (sr, sp) = (np_sum(&wr), np_sum(&wp));
            let a: Vec<f64> = wr.iter().map(|w| w / sr).collect();
            let b: Vec<f64> = wp.iter().map(|w| w / sp).collect();
            Ok(1.0 - emd(&a, &b, &cost))
        })
        .collect()
}

fn kl(p: &[f64], q: &[f64]) -> f64 {
    if p.iter().zip(q).any(|(&a, &b)| a != 0.0 && b == 0.0) {
        return f64::INFINITY;
    }
    let t: Vec<f64> = p
        .iter()
        .zip(q)
        .filter(|(&a, &b)| a != 0.0 && b != 0.0)
        .map(|(a, b)| a * (a / b).ln())
        .collect();
    np_sum(&t)
}

fn trapezoid(x: &[f64], y: &[f64]) -> f64 {
    let t: Vec<f64> = (1..x.len())
        .map(|i| (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0)
        .collect();
    np_sum(&t)
}

fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()); // stable, as NumPy's default for these small arrays
    idx
}

/// MAUVE from two quantised distributions (the step after clustering; as `mauve-text`).
pub fn mauve_from_histograms(
    p: &[f64],
    q: &[f64],
    discretization: usize,
    scaling_factor: f64,
) -> f64 {
    let ws = crate::util::linspace(1e-6, 1.0 - 1e-6, discretization);
    let mut curve = vec![[0.0, f64::INFINITY]];
    for w in ws {
        let r: Vec<f64> = p
            .iter()
            .zip(q)
            .map(|(a, b)| w * a + (1.0 - w) * b)
            .collect();
        curve.push([kl(q, &r), kl(p, &r)]);
    }
    curve.push([f64::INFINITY, 0.0]);
    let x: Vec<f64> = curve
        .iter()
        .map(|c| (-scaling_factor * c[0]).exp())
        .collect();
    let y: Vec<f64> = curve
        .iter()
        .map(|c| (-scaling_factor * c[1]).exp())
        .collect();
    let (i1, i2) = (argsort(&x), argsort(&y));
    let pick = |v: &[f64], i: &[usize]| -> Vec<f64> { i.iter().map(|&k| v[k]).collect() };
    0.5 * (trapezoid(&pick(&x, &i1), &pick(&y, &i1)) + trapezoid(&pick(&y, &i2), &pick(&x, &i2)))
}

/// Options of [`mauve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MauveOpts {
    /// Number of k-means clusters; `None` = max(2, round(min(n_p, n_q) / 10)).
    pub num_buckets: Option<usize>,
    pub explained_variance: f64,
    pub kmeans_n_init: usize,
    pub kmeans_max_iter: usize,
    pub discretization: usize,
    pub scaling_factor: f64,
    /// MAUVE* (Krichevsky–Trofimov smoothing).
    pub smoothed: bool,
    pub seed: u64,
}

impl Default for MauveOpts {
    fn default() -> Self {
        MauveOpts {
            num_buckets: None,
            explained_variance: 0.9,
            kmeans_n_init: 5,
            kmeans_max_iter: 500,
            discretization: 25,
            scaling_factor: 5.0,
            smoothed: false,
            seed: 0,
        }
    }
}

/// Symmetric eigen-decomposition (cyclic Jacobi): eigenvalues descending with their eigenvectors as columns.
fn eigh(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let scale: f64 = (0..n)
            .map(|i| a[i][i] * a[i][i])
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| a[y][y].partial_cmp(&a[x][x]).unwrap());
    let vals = idx.iter().map(|&i| a[i][i].max(0.0)).collect();
    let vecs = (0..n)
        .map(|r| idx.iter().map(|&i| v[r][i]).collect())
        .collect();
    (vals, vecs)
}

fn sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn kmeans(x: &[Vec<f64>], k: usize, seed: u64, n_init: usize, max_iter: usize) -> Vec<usize> {
    let mut rng = evalsuite_core::resample::Xoshiro256pp::new(seed, 0);
    let uniform = |r: &mut evalsuite_core::resample::Xoshiro256pp| {
        r.below(1 << 53) as f64 / (1u64 << 53) as f64
    };
    let n = x.len();
    let (mut best, mut best_inertia) = (vec![0; n], f64::INFINITY);
    for _ in 0..n_init.max(1) {
        let mut centers = vec![x[rng.below(n as u64) as usize].clone()];
        for _ in 1..k {
            let d2: Vec<f64> = x
                .iter()
                .map(|p| {
                    centers
                        .iter()
                        .map(|c| sq(p, c))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let total: f64 = d2.iter().sum();
            let pick = if total > 0.0 {
                let mut u = uniform(&mut rng) * total;
                let mut j = 0;
                while j + 1 < n && u >= d2[j] {
                    u -= d2[j];
                    j += 1;
                }
                j
            } else {
                rng.below(n as u64) as usize
            };
            centers.push(x[pick].clone());
        }
        let mut labels = vec![0usize; n];
        for it in 0..max_iter {
            let new: Vec<usize> = x
                .iter()
                .map(|p| {
                    (0..k)
                        .min_by(|&a, &b| {
                            sq(p, &centers[a]).partial_cmp(&sq(p, &centers[b])).unwrap()
                        })
                        .unwrap()
                })
                .collect();
            if it > 0 && new == labels {
                break;
            }
            labels = new;
            for (j, c) in centers.iter_mut().enumerate() {
                let members: Vec<&Vec<f64>> = x
                    .iter()
                    .zip(&labels)
                    .filter(|(_, &l)| l == j)
                    .map(|(p, _)| p)
                    .collect();
                if !members.is_empty() {
                    for (d, cd) in c.iter_mut().enumerate() {
                        *cd = members.iter().map(|m| m[d]).sum::<f64>() / members.len() as f64;
                    }
                }
            }
        }
        let inertia: f64 = x
            .iter()
            .zip(&labels)
            .map(|(p, &l)| sq(p, &centers[l]))
            .sum();
        if inertia < best_inertia {
            best_inertia = inertia;
            best = labels;
        }
    }
    best
}

/// MAUVE result with the clustering settings used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mauve {
    pub value: f64,
    pub num_buckets: usize,
    pub pca_dims: usize,
}

/// MAUVE between reference and generated text features (e.g. GPT-2 large last-token hidden states): unit
/// rows, PCA to the given explained variance, k-means++ quantisation, then [`mauve_from_histograms`].
///
/// The clustering uses this crate's seeded generator, so values differ slightly from the Python package's
/// (NumPy-seeded) clustering; the histogram stage is identical. Report the seed.
pub fn mauve(
    reference_features: &[Vec<f64>],
    generated_features: &[Vec<f64>],
    o: &MauveOpts,
) -> Result<Mauve> {
    let dp = vectors(reference_features, "reference_features")?;
    let dq = vectors(generated_features, "generated_features")?;
    if dp != dq {
        return invalid("reference_features and generated_features must have the same dimension.");
    }
    if !(o.explained_variance > 0.0 && o.explained_variance < 1.0) {
        return invalid("explained_variance must be in (0, 1).");
    }
    let (np_, nq) = (reference_features.len(), generated_features.len());
    let k = match o.num_buckets {
        None => ((np_.min(nq) as f64 / 10.0).round_ties_even() as usize).max(2),
        Some(k) if k >= 2 => k,
        Some(_) => return invalid("num_buckets must be 'auto' or an integer >= 2."),
    };
    if k > np_ + nq {
        return invalid(format!(
            "num_buckets ({k}) exceeds the number of texts ({}).",
            np_ + nq
        ));
    }
    let mut data = unit(generated_features)?;
    data.extend(unit(reference_features)?);
    let (n, d) = (data.len(), dp);
    let mu: Vec<f64> = (0..d)
        .map(|j| data.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let c: Vec<Vec<f64>> = data
        .iter()
        .map(|r| r.iter().zip(&mu).map(|(x, m)| x - m).collect())
        .collect();
    // Principal components from the smaller Gram matrix: X Xᵀ (n ≤ d) or Xᵀ X.
    let reduced: Vec<Vec<f64>>;
    let dims;
    if n <= d {
        let g: Vec<Vec<f64>> = c
            .iter()
            .map(|a| c.iter().map(|b| dot(a, b)).collect())
            .collect();
        let (vals, vecs) = eigh(g);
        dims = pca_dims(&vals, o.explained_variance);
        // Scores of the first `dims` components: U·S, where U are the eigenvectors of X Xᵀ.
        reduced = (0..n)
            .map(|i| (0..dims).map(|k| vecs[i][k] * vals[k].sqrt()).collect())
            .collect();
    } else {
        let g: Vec<Vec<f64>> = (0..d)
            .map(|a| {
                (0..d)
                    .map(|b| c.iter().map(|r| r[a] * r[b]).sum())
                    .collect()
            })
            .collect();
        let (vals, vecs) = eigh(g);
        dims = pca_dims(&vals, o.explained_variance);
        reduced = c
            .iter()
            .map(|r| {
                (0..dims)
                    .map(|k| (0..d).map(|j| r[j] * vecs[j][k]).sum())
                    .collect()
            })
            .collect();
    }
    let labels = kmeans(&reduced, k, o.seed, o.kmeans_n_init, o.kmeans_max_iter);
    let mut qc = vec![0.0; k];
    let mut pc = vec![0.0; k];
    for (i, &l) in labels.iter().enumerate() {
        if i < nq {
            qc[l] += 1.0;
        } else {
            pc[l] += 1.0;
        }
    }
    if o.smoothed {
        qc.iter_mut().chain(pc.iter_mut()).for_each(|x| *x += 0.5);
    }
    let (sp, sq_) = (pc.iter().sum::<f64>(), qc.iter().sum::<f64>());
    let p: Vec<f64> = pc.iter().map(|x| x / sp).collect();
    let q: Vec<f64> = qc.iter().map(|x| x / sq_).collect();
    Ok(Mauve {
        value: mauve_from_histograms(&p, &q, o.discretization, o.scaling_factor),
        num_buckets: k,
        pca_dims: dims,
    })
}

fn pca_dims(vals: &[f64], explained: f64) -> usize {
    let total: f64 = vals.iter().sum();
    let mut cum = 0.0;
    for (i, v) in vals.iter().enumerate() {
        cum += if total > 0.0 {
            v / total
        } else {
            1.0 / vals.len() as f64
        };
        if cum >= explained {
            return i + 1;
        }
    }
    vals.len()
}

/// Runs a learned metric (`scorer(references, predictions, sources)` → one score per example) in batches and
/// validates its output.
pub fn model_score<R, P, S>(
    references: &[R],
    predictions: &[P],
    sources: Option<&[S]>,
    batch_size: usize,
    scorer: &mut dyn FnMut(&[R], &[P], Option<&[S]>) -> Vec<f64>,
) -> Result<Vec<f64>> {
    if references.is_empty() || references.len() != predictions.len() {
        return invalid("references and predictions must be non-empty and the same length.");
    }
    if sources.is_some_and(|s| s.len() != references.len()) {
        return invalid("sources must have one entry per example.");
    }
    if batch_size == 0 {
        return invalid("batch_size must be a positive integer.");
    }
    let mut out = Vec::with_capacity(references.len());
    for start in (0..references.len()).step_by(batch_size) {
        let end = (start + batch_size).min(references.len());
        let got = scorer(
            &references[start..end],
            &predictions[start..end],
            sources.map(|s| &s[start..end]),
        );
        if got.len() != end - start {
            return invalid(format!(
                "scorer returned {} scores for a batch of {} examples.",
                got.len(),
                end - start
            ));
        }
        if got.iter().any(|v| !v.is_finite()) {
            return invalid("scorer returned NaN or infinite scores.");
        }
        out.extend(got);
    }
    Ok(out)
}
