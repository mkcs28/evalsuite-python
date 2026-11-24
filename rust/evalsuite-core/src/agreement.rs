//! Pairwise-preference kernels: sequential Elo and the Bradley–Terry win matrix, as EvalSuite's Python reference (`text/judge.py`).

/// Sequential Elo ratings. `rows` are (a, b, outcome for a); returns the final ratings of `m` players.
pub fn elo(
    rows: &[(usize, usize, f64)],
    m: usize,
    k: f64,
    initial: f64,
    scale: f64,
    base: f64,
) -> Vec<f64> {
    let mut r = vec![initial; m];
    for &(a, b, o) in rows {
        let ea = 1.0 / (1.0 + base.powf((r[b] - r[a]) / scale));
        r[a] += k * (o - ea);
        r[b] += k * ((1.0 - o) - (1.0 - ea));
    }
    r
}

/// Bradley–Terry win matrix (row-major m × m): wins[a][b] += o and wins[b][a] += 1 − o for every row, in order.
pub fn win_matrix(rows: &[(usize, usize, f64)], m: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; m * m];
    for &(a, b, o) in rows {
        w[a * m + b] += o;
        w[b * m + a] += 1.0 - o;
    }
    w
}

#[cfg(test)]
mod tests {
    #[test]
    fn elo_symmetric() {
        let r = super::elo(&[(0, 1, 1.0)], 2, 4.0, 1000.0, 400.0, 10.0);
        assert_eq!(r, vec![1002.0, 998.0]);
    }
}
