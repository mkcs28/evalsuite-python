//! Surface (boundary) distances between two binary masks of 2 or 3 dimensions, for Hausdorff distance and
//! average symmetric surface distance, as EvalSuite's Python reference (`vision/segmentation.py`): the surface
//! is foreground with a background face neighbour (or on the image border); distances are exact Euclidean
//! nearest-surface distances computed as sqrt(min Σ_k (a_k·s_k − b_k·s_k)²), the same arithmetic as the direct
//! comparison, found with an exact uniform-grid search instead of comparing every pair.

use rayon::prelude::*;

/// Surface points of `mask` (C order, shape `dims`), as index coordinates in C order (as `np.argwhere`).
pub fn surface_points(mask: &[bool], dims: &[usize]) -> Vec<[i64; 3]> {
    let nd = dims.len();
    let mut d3 = [1usize; 3];
    d3[3 - nd..].copy_from_slice(dims);
    let (n0, n1, n2) = (d3[0], d3[1], d3[2]);
    let strides = [n1 * n2, n2, 1];
    // Parallel over rows (slab × row), not only slabs, so 2-D images use every core; rows are collected in
    // order, so the points come out in the same (C) order as a serial scan.
    let per_slab: Vec<Vec<[i64; 3]>> = (0..n0 * n1)
        .into_par_iter()
        .with_min_len(16)
        .map(|r| {
            let (i, j) = (r / n1, r % n1);
            let mut out = Vec::new();
            let row = r * n2;
            for k in 0..n2 {
                let f = row + k;
                if !mask[f] {
                    continue;
                }
                let idx = [i, j, k];
                // Surface: on the image border along a real axis, or a face neighbour is background.
                let border = (3 - nd..3).any(|ax| {
                    let p = idx[ax];
                    p == 0 || p + 1 == d3[ax] || !mask[f - strides[ax]] || !mask[f + strides[ax]]
                });
                if border {
                    out.push([i as i64, j as i64, k as i64]);
                }
            }
            out
        })
        .collect();
    per_slab.concat()
}

#[inline]
fn dist2(a: &[i64; 3], b: &[i64; 3], s: &[f64; 3], first: usize) -> f64 {
    let mut d = 0.0f64;
    for k in first..3 {
        let x = a[k] as f64 * s[k] - b[k] as f64 * s[k];
        d += x * x;
    }
    d
}

/// A k-d tree over points (leaf buckets of up to 16), split at the median of the widest axis, with the tight
/// bounding box of every node for pruning.
struct KdTree<'a> {
    pts: &'a [[i64; 3]],
    idx: Vec<usize>,
    nodes: Vec<Node>,
}

struct Node {
    lo: [i64; 3],
    hi: [i64; 3],
    /// Leaf: the points `idx[start..end]`; inner node: `children`.
    start: usize,
    end: usize,
    children: Option<(usize, usize)>,
}

#[allow(clippy::needless_range_loop)] // axes index several parallel arrays
impl<'a> KdTree<'a> {
    fn new(pts: &'a [[i64; 3]], first: usize) -> Self {
        let mut t = KdTree {
            pts,
            idx: (0..pts.len()).collect(),
            nodes: Vec::new(),
        };
        let n = pts.len();
        t.build(0, n, first);
        t
    }

    fn build(&mut self, start: usize, end: usize, first: usize) -> usize {
        let pts = self.pts;
        let mut lo = [i64::MAX; 3];
        let mut hi = [i64::MIN; 3];
        for &i in &self.idx[start..end] {
            for a in 0..3 {
                lo[a] = lo[a].min(pts[i][a]);
                hi[a] = hi[a].max(pts[i][a]);
            }
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            lo,
            hi,
            start,
            end,
            children: None,
        });
        let (mut axis, mut width) = (first, 0i64);
        for a in first..3 {
            if hi[a] - lo[a] > width {
                width = hi[a] - lo[a];
                axis = a;
            }
        }
        if end - start <= 16 || width == 0 {
            return id;
        }
        let mid = (start + end) / 2;
        self.idx[start..end].select_nth_unstable_by_key(mid - start, |&i| pts[i][axis]);
        let left = self.build(start, mid, first);
        let right = self.build(mid, end, first);
        self.nodes[id].children = Some((left, right));
        id
    }

    /// A lower bound on dist2 from `q` to any point in node `n`'s box (the same per-axis arithmetic).
    fn box_bound(&self, n: usize, q: &[i64; 3], s: &[f64; 3], first: usize) -> f64 {
        let node = &self.nodes[n];
        let mut d = 0.0f64;
        for k in first..3 {
            let c = q[k].clamp(node.lo[k], node.hi[k]);
            let x = q[k] as f64 * s[k] - c as f64 * s[k];
            d += x * x;
        }
        d
    }

    /// Smallest dist2 from `q` to any point (exact: the same arithmetic as comparing every point).
    fn nearest(&self, q: &[i64; 3], s: &[f64; 3], first: usize, exact: bool) -> f64 {
        let mut best = f64::INFINITY;
        let mut stack: Vec<(usize, f64)> = vec![(0, self.box_bound(0, q, s, first))];
        while let Some((n, bound)) = stack.pop() {
            // With unit spacing every bound and distance is an exact integer, so a node that cannot beat the best
            // is pruned at equality; otherwise a safety margin keeps pruning conservative against rounding.
            if (exact && bound >= best) || bound * (1.0 - 1e-9) > best {
                continue;
            }
            let node = &self.nodes[n];
            match node.children {
                None => {
                    for &i in &self.idx[node.start..node.end] {
                        let d = dist2(q, &self.pts[i], s, first);
                        if d < best {
                            best = d;
                        }
                    }
                }
                Some((l, r)) => {
                    let (bl, br) = (
                        self.box_bound(l, q, s, first),
                        self.box_bound(r, q, s, first),
                    );
                    // Visit the nearer child first (pushed last).
                    if bl <= br {
                        stack.push((r, br));
                        stack.push((l, bl));
                    } else {
                        stack.push((l, bl));
                        stack.push((r, br));
                    }
                }
            }
        }
        best
    }
}

/// For each point of `src`, the distance to its nearest point of `dst` (non-empty), in an image of shape
/// `dims`. `spacing` per real axis. Nearby points are found by scanning a dense bitmap of `dst` in growing
/// Chebyshev shells, stopping once no unvisited point can be closer; distant ones fall back to the k-d tree.
/// Either way the result is the minimum of the same per-pair arithmetic over a set containing the nearest
/// point, so it equals comparing every pair.
pub fn nearest(src: &[[i64; 3]], dst: &[[i64; 3]], dims: &[usize], spacing: &[f64]) -> Vec<f64> {
    const SHELLS: i64 = 3;
    let nd = dims.len();
    let mut s = [1.0f64; 3];
    s[3 - nd..].copy_from_slice(spacing);
    let first = 3 - nd;
    let mut d3 = [1i64; 3];
    for (k, &d) in dims.iter().enumerate() {
        d3[3 - nd + k] = d as i64;
    }
    let mut occupied = vec![false; (d3[0] * d3[1] * d3[2]) as usize];
    for p in dst {
        occupied[((p[0] * d3[1] + p[1]) * d3[2] + p[2]) as usize] = true;
    }
    let tree = KdTree::new(dst, first);
    // Unit spacing and coordinates below 2^26: squared distances are exact integers in f64.
    let exact = spacing.iter().all(|&x| x == 1.0)
        && src
            .iter()
            .chain(dst)
            .all(|p| p.iter().all(|&c| c < (1 << 26)));
    let min_s = spacing.iter().copied().fold(f64::INFINITY, f64::min);
    src.par_iter()
        .with_min_len(256)
        .map(|q| {
            let mut best = f64::INFINITY;
            for r in 0..=SHELLS {
                let span = |k: usize| -> (i64, i64) {
                    if k < first {
                        (0, 0)
                    } else {
                        ((q[k] - r).max(0), (q[k] + r).min(d3[k] - 1))
                    }
                };
                let (a0, a1, a2) = (span(0), span(1), span(2));
                for x in a0.0..=a0.1 {
                    for y in a1.0..=a1.1 {
                        let row = (x * d3[1] + y) * d3[2];
                        let edge_xy = (x - q[0]).abs() == r || (y - q[1]).abs() == r;
                        // Only the shell: interior z cells were visited at a smaller r unless x or y is on it.
                        let mut z = a2.0;
                        while z <= a2.1 {
                            if edge_xy || (z - q[2]).abs() == r {
                                if occupied[(row + z) as usize] {
                                    let d = dist2(q, &[x, y, z], &s, first);
                                    if d < best {
                                        best = d;
                                    }
                                }
                                z += 1;
                            } else {
                                z = q[2] + r; // jump across the interior to the far face
                            }
                        }
                    }
                }
                // Unvisited points are at least r + 1 index steps away along some axis.
                let lb = (r + 1) as f64 * min_s;
                let lb2 = lb * lb;
                if (exact && best <= lb2) || best < lb2 * (1.0 - 1e-9) {
                    return best.sqrt();
                }
            }
            tree.nearest(q, &s, first, exact).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_surface_and_distances() {
        // 4×4 mask with a 3×3 block: its 8 border pixels are surface, the centre is not.
        let mut m = vec![false; 16];
        for i in 0..3 {
            for j in 0..3 {
                m[i * 4 + j] = true;
            }
        }
        let s = surface_points(&m, &[4, 4]);
        assert_eq!(s.len(), 8);
        let d = nearest(&[[0, 0, 0]], &[[0, 3, 4]], &[5, 5], &[1.0, 1.0]);
        assert_eq!(d, vec![5.0]);
    }

    #[test]
    fn grid_matches_brute_force() {
        let pts: Vec<[i64; 3]> = (0..400)
            .map(|i| [0, (i * 37) % 101, (i * 53) % 89])
            .collect();
        let q: Vec<[i64; 3]> = (0..300)
            .map(|i| [0, (i * 11) % 120, (i * 7) % 95])
            .collect();
        let sp = [0.7, 1.3];
        let got = nearest(&q, &pts, &[120, 101], &sp);
        for (a, g) in q.iter().zip(&got) {
            let s = [1.0, 0.7, 1.3];
            let want = pts
                .iter()
                .map(|b| dist2(a, b, &s, 1))
                .fold(f64::INFINITY, f64::min)
                .sqrt();
            assert_eq!(*g, want);
        }
    }
}
