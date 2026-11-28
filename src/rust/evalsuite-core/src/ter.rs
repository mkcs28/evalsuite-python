//! Translation edit rate: Tercom's beam edit distance with block shifts, exactly as sacreBLEU (and EvalSuite's
//! Python reference `_ter_edits`). Words are integer ids.

const MAX_SHIFT_SIZE: usize = 10;
const MAX_SHIFT_DIST: usize = 50;
const BEAM_WIDTH: i64 = 25;
const MAX_SHIFT_CANDIDATES: usize = 1000;
const INF: i64 = 10_000_000_000_000_000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Match,
    Sub,
    Del,
    Ins,
    None,
}

/// (gain, length, -start, -target): Tercom's preference order for shift candidates, compared lexicographically.
type ShiftKey = (i64, usize, i64, i64);

fn beam(nh: usize, nr: usize) -> (f64, i64) {
    let ratio = if nh > 0 { nr as f64 / nh as f64 } else { 1.0 };
    let b = if ratio / 2.0 > BEAM_WIDTH as f64 {
        (ratio / 2.0 + BEAM_WIDTH as f64).ceil() as i64
    } else {
        BEAM_WIDTH
    };
    (ratio, b)
}

/// Fill rows `start..=nh` of the beam edit-distance table (`rows` holds rows `0..start`, each `nr + 1` wide).
/// With `ops`, also record each cell's operation (ties broken match/substitute, delete, insert).
fn edit_rows(
    hyp: &[u32],
    reference: &[u32],
    rows: &mut Vec<Vec<i64>>,
    start: usize,
    mut ops: Option<&mut Vec<Vec<Op>>>,
) {
    let (nh, nr) = (hyp.len(), reference.len());
    let (ratio, b) = beam(nh, nr);
    rows.truncate(start.max(1));
    for i in start.max(1)..=nh {
        let diag = (i as f64 * ratio).floor() as i64;
        let lo = (diag - b).max(0) as usize;
        let hi = if i == nh {
            nr + 1
        } else {
            ((diag + b).max(0) as usize).min(nr + 1)
        };
        let mut row = vec![INF; nr + 1];
        let mut oprow = if ops.is_some() {
            Some(vec![Op::None; nr + 1])
        } else {
            None
        };
        {
            let prev = &rows[i - 1];
            let w = hyp[i - 1];
            for j in lo..hi {
                if j == 0 {
                    row[0] = prev[0] + 1;
                    if let Some(o) = oprow.as_mut() {
                        o[0] = Op::Del;
                    }
                    continue;
                }
                let is_match = w == reference[j - 1];
                let mut best = prev[j - 1] + i64::from(!is_match);
                let mut op = if is_match { Op::Match } else { Op::Sub };
                let d = prev[j] + 1;
                if d < best {
                    best = d;
                    op = Op::Del;
                }
                let ins = row[j - 1] + 1;
                if ins < best {
                    best = ins;
                    op = Op::Ins;
                }
                if best < INF {
                    row[j] = best;
                    if let Some(o) = oprow.as_mut() {
                        o[j] = op;
                    }
                }
            }
        }
        rows.push(row);
        if let (Some(o), Some(all)) = (oprow, ops.as_mut()) {
            all.truncate(i);
            all.push(o);
        }
    }
}

/// Final cost only, continuing from `base[keep]` (rows `0..=keep` of a hypothesis with the same prefix).
fn edit_cost_from(
    hyp: &[u32],
    reference: &[u32],
    base: &[Vec<i64>],
    keep: usize,
    scratch: &mut Vec<Vec<i64>>,
) -> i64 {
    scratch.clear();
    scratch.extend(base[..=keep].iter().cloned());
    edit_rows(hyp, reference, scratch, keep + 1, None);
    scratch[hyp.len()][reference.len()]
}

/// Edit distance, the cost rows and the trace (in Tercom's convention, before the i/d swap).
fn edit_distance(hyp: &[u32], reference: &[u32]) -> (i64, Vec<Vec<i64>>, Vec<Op>) {
    let (nh, nr) = (hyp.len(), reference.len());
    let mut rows: Vec<Vec<i64>> = vec![(0..=nr as i64).collect()];
    let mut ops: Vec<Vec<Op>> = vec![vec![Op::Ins; nr + 1]];
    edit_rows(hyp, reference, &mut rows, 1, Some(&mut ops));
    let mut trace = Vec::with_capacity(nh + nr);
    let (mut i, mut j) = (nh, nr);
    while i > 0 || j > 0 {
        let op = ops[i][j];
        trace.push(op);
        match op {
            Op::Sub | Op::Match => {
                i -= 1;
                j -= 1;
            }
            Op::Ins => j -= 1,
            Op::Del => i -= 1,
            Op::None => unreachable!("invalid edit trace"),
        }
    }
    trace.reverse();
    (rows[nh][nr], rows, trace)
}

fn perform_shift(words: &[u32], start: usize, length: usize, target: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(words.len());
    if target < start {
        out.extend_from_slice(&words[..target]);
        out.extend_from_slice(&words[start..start + length]);
        out.extend_from_slice(&words[target..start]);
        out.extend_from_slice(&words[start + length..]);
    } else if target > start + length {
        out.extend_from_slice(&words[..start]);
        out.extend_from_slice(&words[start + length..target]);
        out.extend_from_slice(&words[start..start + length]);
        out.extend_from_slice(&words[target..]);
    } else {
        // Python slices clamp at the end of the list; mirror that.
        let n = words.len();
        let a = (length + target).min(n);
        out.extend_from_slice(&words[..start]);
        out.extend_from_slice(&words[(start + length).min(a)..a]);
        out.extend_from_slice(&words[start..start + length]);
        out.extend_from_slice(&words[a..]);
    }
    out
}

/// One round of shifting: (gain, shifted words, candidates checked, edit distance of `hyp`).
fn ter_shift(
    hyp: &[u32],
    reference: &[u32],
    mut checked: usize,
) -> (i64, Option<Vec<u32>>, usize, i64) {
    let (pre, base, inv_trace) = edit_distance(hyp, reference);
    let (nh, nr) = (hyp.len(), reference.len());
    // align[pr] = ph for every reference position; error flags per position.
    let mut align: Vec<i64> = vec![i64::MIN; nr];
    let mut ref_err: Vec<u8> = Vec::with_capacity(nr);
    let mut hyp_err: Vec<u8> = Vec::with_capacity(nh);
    let (mut ph, mut pr) = (-1i64, -1i64);
    for op in inv_trace {
        // Tercom swaps insertions and deletions relative to the table's convention.
        let op = match op {
            Op::Ins => Op::Del,
            Op::Del => Op::Ins,
            o => o,
        };
        match op {
            Op::Match | Op::Sub => {
                ph += 1;
                pr += 1;
                align[pr as usize] = ph;
                let e = u8::from(op == Op::Sub);
                hyp_err.push(e);
                ref_err.push(e);
            }
            Op::Ins => {
                ph += 1;
                hyp_err.push(1);
            }
            _ => {
                pr += 1;
                align[pr as usize] = ph;
                ref_err.push(1);
            }
        }
    }
    let sum_slice = |v: &[u8], a: usize, len: usize| -> u32 {
        let end = (a + len).min(v.len());
        if a >= end {
            0
        } else {
            v[a..end].iter().map(|&x| u32::from(x)).sum()
        }
    };
    let mut scratch: Vec<Vec<i64>> = Vec::new();
    // best = (gain, length, -sh, -idx) compared lexicographically, plus the shifted words.
    let mut best: Option<(ShiftKey, Vec<u32>)> = None;
    'outer: for sh in 0..nh {
        let lo = sh.saturating_sub(MAX_SHIFT_DIST);
        let hi = nr.min(sh + MAX_SHIFT_DIST + 1);
        for sr in lo..hi {
            let mut length = 0usize;
            while hyp[sh + length] == reference[sr + length] && length < MAX_SHIFT_SIZE {
                length += 1;
                // ---- one (sh, sr, length) candidate run
                let skip = sum_slice(&hyp_err, sh, length) == 0
                    || sum_slice(&ref_err, sr, length) == 0
                    || {
                        let a = align[sr];
                        (sh as i64) <= a && a < (sh + length) as i64
                    };
                if !skip {
                    let mut prev_idx: i64 = -1;
                    for off in -1i64..length as i64 {
                        let pos = sr as i64 + off;
                        let idx: i64 = if pos == -1 {
                            0
                        } else {
                            align[pos as usize] + 1
                        };
                        if idx == prev_idx {
                            continue;
                        }
                        prev_idx = idx;
                        let idx_u = idx as usize;
                        let shifted = perform_shift(hyp, sh, length, idx_u);
                        let keep = sh.min(idx_u);
                        let cost = edit_cost_from(&shifted, reference, &base, keep, &mut scratch);
                        let key = (pre - cost, length, -(sh as i64), -idx);
                        checked += 1;
                        if best.as_ref().is_none_or(|(k, _)| key > *k) {
                            best = Some((key, shifted));
                        }
                    }
                }
                if checked >= MAX_SHIFT_CANDIDATES {
                    break 'outer;
                }
                // ---- end of candidate
                if nh == sh + length || nr == sr + length {
                    break;
                }
            }
        }
    }
    match best {
        None => (0, None, checked, pre),
        Some(((gain, _, _, _), words)) => (gain, Some(words), checked, pre),
    }
}

/// Number of TER edits (word edits plus block shifts) to turn `hyp` into `reference`.
pub fn ter_edits(hyp: &[u32], reference: &[u32]) -> i64 {
    if reference.is_empty() {
        return hyp.len() as i64;
    }
    let mut shifts = 0i64;
    let mut words: Vec<u32> = hyp.to_vec();
    let mut checked = 0usize;
    loop {
        let (delta, new_words, c, dist) = ter_shift(&words, reference, checked);
        checked = c;
        if checked >= MAX_SHIFT_CANDIDATES || delta <= 0 {
            return shifts + dist;
        }
        shifts += 1;
        words = new_words.expect("a positive gain always comes with a shift");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_zero() {
        assert_eq!(ter_edits(&[1, 2, 3], &[1, 2, 3]), 0);
    }

    #[test]
    fn empty_reference() {
        assert_eq!(ter_edits(&[1, 2], &[]), 2);
    }

    #[test]
    fn shift_counts_one() {
        // "a b c d" -> "c d a b": one block shift of "c d" (or "a b").
        assert_eq!(ter_edits(&[1, 2, 3, 4], &[3, 4, 1, 2]), 1);
    }
}
