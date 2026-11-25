//! Semantic-segmentation metrics on integer label masks, as EvalSuite's Python
//! `evalsuite.vision.segmentation`: Dice, IoU (and mIoU), pixel accuracies, Boundary IoU, Hausdorff
//! distance / HD95 and average symmetric surface distance.
//!
//! Each image is a [`Mask`] (shape and row-major labels). Pixels equal to `ignore_index` are excluded. A
//! class absent from both truth and prediction has an undefined (NaN) score and is left out of the macro
//! average unless `empty` gives it a value.

use crate::error::{invalid, Result};
use crate::util::{np_sum, percentile, sorted};

/// One label mask: `shape` (2 or more axes) and its labels in row-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    pub shape: Vec<usize>,
    pub labels: Vec<i64>,
}

impl Mask {
    pub fn new(shape: Vec<usize>, labels: Vec<i64>) -> Result<Self> {
        if shape.len() < 2 {
            return invalid("A mask must have at least 2 dimensions.");
        }
        if shape.iter().product::<usize>() != labels.len() {
            return invalid("The number of labels does not match the mask shape.");
        }
        Ok(Mask { shape, labels })
    }

    /// A 2-D mask from rows.
    pub fn from_rows(rows: &[Vec<i64>]) -> Result<Self> {
        let w = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != w) {
            return invalid("Every row of a mask must have the same length.");
        }
        Mask::new(vec![rows.len(), w], rows.concat())
    }
}

/// How per-class values are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Average {
    #[default]
    Macro,
    Micro,
    Weighted,
    None,
}

/// Pool pixel counts over all images (`Dataset`, segmentation benchmarks) or average per-image scores
/// (`Image`, medical imaging).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregate {
    #[default]
    Dataset,
    Image,
}

/// Options shared by the overlap metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct Opts {
    pub num_classes: Option<usize>,
    pub labels: Option<Vec<i64>>,
    pub ignore_index: Option<i64>,
    pub average: Average,
    pub aggregate: Aggregate,
    pub include_background: bool,
    /// Score for a class absent from both masks: `None` leaves it out (NaN), `Some(v)` uses v.
    pub empty: Option<f64>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            num_classes: None,
            labels: None,
            ignore_index: None,
            average: Average::Macro,
            aggregate: Aggregate::Dataset,
            include_background: true,
            empty: None,
        }
    }
}

pub use crate::classification::Score;

fn pairs<'a>(t: &'a [Mask], p: &'a [Mask]) -> Result<()> {
    if t.is_empty() {
        return invalid("y_true contains no images.");
    }
    if t.len() != p.len() {
        return invalid(format!(
            "y_true has {} images but y_pred has {}.",
            t.len(),
            p.len()
        ));
    }
    for (i, (a, b)) in t.iter().zip(p).enumerate() {
        if a.shape != b.shape {
            return invalid(format!(
                "Image {i}: y_true has shape {:?} but y_pred has shape {:?}.",
                a.shape, b.shape
            ));
        }
    }
    Ok(())
}

fn classes(
    t: &[Mask],
    p: &[Mask],
    num_classes: Option<usize>,
    labels: Option<&[i64]>,
    ignore: Option<i64>,
    background: bool,
) -> Result<Vec<i64>> {
    let valid = |m: &Mask| {
        m.labels
            .iter()
            .copied()
            .filter(move |&v| Some(v) != ignore)
            .collect::<Vec<_>>()
    };
    let mut cls: Vec<i64> = match labels {
        Some(l) => {
            let mut u = l.to_vec();
            u.sort_unstable();
            u.dedup();
            if l.is_empty() || u.len() != l.len() {
                return invalid("labels must be a non-empty list of distinct class indices.");
            }
            l.to_vec()
        }
        None => {
            let k = match num_classes {
                Some(k) => k as i64,
                None => t.iter().chain(p).flat_map(valid).max().map_or(0, |m| m + 1),
            };
            if k < 1 {
                return invalid("num_classes must be at least 1.");
            }
            (0..k).collect()
        }
    };
    if !background {
        cls.retain(|&c| c != 0);
        if cls.is_empty() {
            return invalid("include_background=false leaves no classes to evaluate.");
        }
    }
    for m in t.iter().chain(p) {
        let v = valid(m);
        if v.iter().any(|&x| x < 0) {
            return invalid("Masks contain negative labels; use ignore_index for void pixels.");
        }
        if let (Some(k), None) = (num_classes, labels) {
            if v.iter().any(|&x| x >= k as i64) {
                return invalid(format!("Masks contain a label >= num_classes={k}."));
            }
        }
    }
    Ok(cls)
}

/// Per image and class: (tp, fp, fn).
fn counts(t: &[Mask], p: &[Mask], cls: &[i64], ignore: Option<i64>) -> Vec<Vec<(f64, f64, f64)>> {
    t.iter()
        .zip(p)
        .map(|(a, b)| {
            cls.iter()
                .map(|&c| {
                    let (mut tp, mut fp, mut fn_) = (0u64, 0u64, 0u64);
                    for (&x, &y) in a.labels.iter().zip(&b.labels) {
                        if Some(x) == ignore {
                            continue;
                        }
                        match (x == c, y == c) {
                            (true, true) => tp += 1,
                            (false, true) => fp += 1,
                            (true, false) => fn_ += 1,
                            _ => {}
                        }
                    }
                    (tp as f64, fp as f64, fn_ as f64)
                })
                .collect()
        })
        .collect()
}

fn score(tp: f64, fp: f64, fn_: f64, dice: bool) -> f64 {
    let (num, den) = if dice {
        (2.0 * tp, 2.0 * tp + fp + fn_)
    } else {
        (tp, tp + fp + fn_)
    };
    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

fn nanmean(v: &[f64]) -> f64 {
    let ok: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if ok.is_empty() {
        f64::NAN
    } else {
        np_sum(&ok) / ok.len() as f64
    }
}

fn reduce(
    mut per_class: Vec<f64>,
    support: &[f64],
    average: Average,
    empty: Option<f64>,
) -> Result<Score> {
    if let Some(e) = empty {
        if !(0.0..=1.0).contains(&e) {
            return invalid("empty must be None or a value in [0, 1] (usually 1.0).");
        }
        for v in per_class.iter_mut() {
            if v.is_nan() {
                *v = e;
            }
        }
    }
    Ok(match average {
        Average::None => Score::PerClass(per_class),
        Average::Macro => Score::Scalar(nanmean(&per_class)),
        Average::Weighted => {
            let ok: Vec<usize> = (0..per_class.len())
                .filter(|&i| !per_class[i].is_nan())
                .collect();
            let w: Vec<f64> = ok.iter().map(|&i| support[i]).collect();
            if np_sum(&w) == 0.0 {
                Score::Scalar(f64::NAN)
            } else {
                Score::Scalar(crate::util::weighted_mean(
                    &ok.iter().map(|&i| per_class[i]).collect::<Vec<_>>(),
                    &w,
                ))
            }
        }
        Average::Micro => unreachable!("handled by the caller"),
    })
}

fn overlap(t: &[Mask], p: &[Mask], o: &Opts, dice: bool) -> Result<Score> {
    pairs(t, p)?;
    let cls = classes(
        t,
        p,
        o.num_classes,
        o.labels.as_deref(),
        o.ignore_index,
        o.include_background,
    )?;
    let c = counts(t, p, &cls, o.ignore_index);
    let k = cls.len();
    let sum_img = |f: fn(&(f64, f64, f64)) -> f64| -> Vec<f64> {
        (0..k)
            .map(|j| c.iter().map(|img| f(&img[j])).sum())
            .collect()
    };
    let (tp, fp, fn_) = (sum_img(|x| x.0), sum_img(|x| x.1), sum_img(|x| x.2));
    if o.average == Average::Micro {
        return Ok(Score::Scalar(match o.aggregate {
            Aggregate::Dataset => score(np_sum(&tp), np_sum(&fp), np_sum(&fn_), dice),
            Aggregate::Image => nanmean(
                &c.iter()
                    .map(|img| {
                        score(
                            img.iter().map(|x| x.0).sum(),
                            img.iter().map(|x| x.1).sum(),
                            img.iter().map(|x| x.2).sum(),
                            dice,
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
        }));
    }
    let support: Vec<f64> = tp.iter().zip(&fn_).map(|(a, b)| a + b).collect();
    let per_class: Vec<f64> = match o.aggregate {
        Aggregate::Dataset => (0..k).map(|j| score(tp[j], fp[j], fn_[j], dice)).collect(),
        Aggregate::Image => (0..k)
            .map(|j| {
                let s: Vec<f64> = c
                    .iter()
                    .map(|img| {
                        let v = score(img[j].0, img[j].1, img[j].2, dice);
                        if v.is_nan() {
                            o.empty.unwrap_or(f64::NAN)
                        } else {
                            v
                        }
                    })
                    .collect();
                nanmean(&s)
            })
            .collect(),
    };
    reduce(per_class, &support, o.average, o.empty)
}

/// Dice coefficient (F1 of pixels) per class.
pub fn dice(y_true: &[Mask], y_pred: &[Mask], o: &Opts) -> Result<Score> {
    overlap(y_true, y_pred, o, true)
}

/// Intersection over union (Jaccard) per class.
pub fn iou(y_true: &[Mask], y_pred: &[Mask], o: &Opts) -> Result<Score> {
    overlap(y_true, y_pred, o, false)
}

/// Mean IoU over classes (dataset aggregation).
pub fn miou(
    y_true: &[Mask],
    y_pred: &[Mask],
    num_classes: Option<usize>,
    ignore_index: Option<i64>,
    include_background: bool,
) -> Result<f64> {
    let o = Opts {
        num_classes,
        ignore_index,
        include_background,
        ..Default::default()
    };
    Ok(iou(y_true, y_pred, &o)?.value().unwrap())
}

/// Share of (non-ignored) pixels labelled correctly.
pub fn pixel_accuracy(y_true: &[Mask], y_pred: &[Mask], ignore_index: Option<i64>) -> Result<f64> {
    pairs(y_true, y_pred)?;
    let (mut correct, mut total) = (0u64, 0u64);
    for (a, b) in y_true.iter().zip(y_pred) {
        for (&x, &y) in a.labels.iter().zip(&b.labels) {
            if Some(x) != ignore_index {
                total += 1;
                correct += u64::from(x == y);
            }
        }
    }
    if total == 0 {
        return invalid("Every pixel is ignored; nothing to evaluate.");
    }
    Ok(correct as f64 / total as f64)
}

/// Mean over classes present in the truth of the per-class pixel recall.
pub fn mean_pixel_accuracy(
    y_true: &[Mask],
    y_pred: &[Mask],
    num_classes: Option<usize>,
    ignore_index: Option<i64>,
) -> Result<f64> {
    pairs(y_true, y_pred)?;
    let cls = classes(y_true, y_pred, num_classes, None, ignore_index, true)?;
    let c = counts(y_true, y_pred, &cls, ignore_index);
    let r: Vec<f64> = (0..cls.len())
        .filter_map(|j| {
            let t: f64 = c.iter().map(|img| img[j].0).sum();
            let f: f64 = c.iter().map(|img| img[j].2).sum();
            (t + f > 0.0).then(|| t / (t + f))
        })
        .collect();
    Ok(np_sum(&r) / r.len() as f64)
}

/// Erosion of a binary mask by a (2d+1)^n cube with the outside counted as background.
fn erode_cube(mask: &[bool], shape: &[usize], d: usize) -> Vec<bool> {
    let mut cur = mask.to_vec();
    let n = shape.len();
    for ax in 0..n {
        let stride: usize = shape[ax + 1..].iter().product();
        let len = shape[ax];
        let mut next = vec![false; cur.len()];
        for (i, out) in next.iter_mut().enumerate() {
            let pos = (i / stride) % len;
            if pos < d || pos + d >= len {
                continue;
            }
            let base = i - pos * stride;
            *out = (pos - d..=pos + d).all(|q| cur[base + q * stride]);
        }
        cur = next;
    }
    cur
}

/// Boundary IoU (Cheng et al. 2021) per class, with band width d = max(1, round(dilation_ratio × diagonal)).
pub fn boundary_iou(
    y_true: &[Mask],
    y_pred: &[Mask],
    dilation_ratio: f64,
    o: &Opts,
) -> Result<Score> {
    if !(dilation_ratio > 0.0 && dilation_ratio < 1.0) {
        return invalid("dilation_ratio must be between 0 and 1 (0.02 is the paper's default).");
    }
    pairs(y_true, y_pred)?;
    let cls = classes(
        y_true,
        y_pred,
        o.num_classes,
        o.labels.as_deref(),
        o.ignore_index,
        o.include_background,
    )?;
    let mut inter = vec![0.0f64; cls.len()];
    let mut union = vec![0.0f64; cls.len()];
    for (a, b) in y_true.iter().zip(y_pred) {
        let diag = (a.shape.iter().map(|&s| (s * s) as f64).sum::<f64>()).sqrt();
        let d = ((dilation_ratio * diag).round_ties_even() as usize).max(1);
        let valid: Vec<bool> = a
            .labels
            .iter()
            .map(|&x| Some(x) != o.ignore_index)
            .collect();
        for (j, &c) in cls.iter().enumerate() {
            let band = |m: &Mask| -> Vec<bool> {
                let bin: Vec<bool> = m
                    .labels
                    .iter()
                    .zip(&valid)
                    .map(|(&x, &v)| x == c && v)
                    .collect();
                if !bin.iter().any(|&x| x) {
                    return bin;
                }
                let er = erode_cube(&bin, &m.shape, d);
                bin.iter().zip(&er).map(|(&x, &e)| x && !e).collect()
            };
            let (gb, pb) = (band(a), band(b));
            inter[j] += gb.iter().zip(&pb).filter(|(x, y)| **x && **y).count() as f64;
            union[j] += gb.iter().zip(&pb).filter(|(x, y)| **x || **y).count() as f64;
        }
    }
    if o.average == Average::Micro {
        return Ok(Score::Scalar(np_sum(&inter) / np_sum(&union)));
    }
    let per_class: Vec<f64> = inter
        .iter()
        .zip(&union)
        .map(|(i, u)| if *u > 0.0 { i / u } else { f64::NAN })
        .collect();
    reduce(per_class, &union, o.average, o.empty)
}

/// Options for the surface-distance metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceOpts {
    /// Physical size of a pixel along each axis (default 1).
    pub spacing: Option<Vec<f64>>,
    pub num_classes: Option<usize>,
    pub labels: Option<Vec<i64>>,
    pub ignore_index: Option<i64>,
    pub include_background: bool,
    /// Macro average over classes (`true`) or per-class values.
    pub macro_average: bool,
}

impl Default for SurfaceOpts {
    fn default() -> Self {
        SurfaceOpts {
            spacing: None,
            num_classes: None,
            labels: None,
            ignore_index: None,
            include_background: false,
            macro_average: true,
        }
    }
}

fn surface_metric(
    t: &[Mask],
    p: &[Mask],
    o: &SurfaceOpts,
    f: &dyn Fn(&[f64], &[f64]) -> f64,
) -> Result<Score> {
    pairs(t, p)?;
    let cls = classes(
        t,
        p,
        o.num_classes,
        o.labels.as_deref(),
        o.ignore_index,
        o.include_background,
    )?;
    let mut values = vec![vec![f64::NAN; cls.len()]; t.len()];
    for (i, (a, b)) in t.iter().zip(p).enumerate() {
        let nd = a.shape.len();
        if !(2..=3).contains(&nd) {
            return invalid("Surface distances support 2-D and 3-D masks.");
        }
        let sp = match &o.spacing {
            None => vec![1.0; nd],
            Some(s) => {
                if s.len() != nd || s.iter().any(|&x| !(x > 0.0 && x.is_finite())) {
                    return invalid(format!(
                        "spacing must have {nd} positive values (one per image axis)."
                    ));
                }
                s.clone()
            }
        };
        for (j, &c) in cls.iter().enumerate() {
            let bin = |m: &Mask| -> Vec<bool> {
                m.labels
                    .iter()
                    .zip(&a.labels)
                    .map(|(&x, &v)| x == c && Some(v) != o.ignore_index)
                    .collect()
            };
            let (ta, pb) = (bin(a), bin(b));
            let (ha, hb) = (ta.iter().any(|&x| x), pb.iter().any(|&x| x));
            if !ha && !hb {
                continue;
            }
            if !ha || !hb {
                values[i][j] = f64::INFINITY;
                continue;
            }
            let sa = evalsuite_core::surface::surface_points(&ta, &a.shape);
            let sb = evalsuite_core::surface::surface_points(&pb, &a.shape);
            let dab = evalsuite_core::surface::nearest(&sa, &sb, &a.shape, &sp);
            let dba = evalsuite_core::surface::nearest(&sb, &sa, &a.shape, &sp);
            values[i][j] = f(&dab, &dba);
        }
    }
    let per_class: Vec<f64> = (0..cls.len())
        .map(|j| nanmean(&values.iter().map(|r| r[j]).collect::<Vec<_>>()))
        .collect();
    if !o.macro_average {
        return Ok(Score::PerClass(per_class));
    }
    Ok(Score::Scalar(nanmean(&per_class)))
}

/// Hausdorff distance between the class boundaries (`percentile = 95` gives HD95), per class; infinite when
/// the class occurs in only one of the two masks.
pub fn hausdorff_distance(
    y_true: &[Mask],
    y_pred: &[Mask],
    percentile_q: f64,
    o: &SurfaceOpts,
) -> Result<Score> {
    if !(percentile_q > 0.0 && percentile_q <= 100.0) {
        return invalid("percentile must be in (0, 100].");
    }
    surface_metric(y_true, y_pred, o, &|a, b| {
        if percentile_q == 100.0 {
            a.iter().chain(b).copied().fold(f64::NEG_INFINITY, f64::max)
        } else {
            percentile(&sorted(a), percentile_q).max(percentile(&sorted(b), percentile_q))
        }
    })
}

/// Average symmetric surface distance, per class.
pub fn average_surface_distance(
    y_true: &[Mask],
    y_pred: &[Mask],
    o: &SurfaceOpts,
) -> Result<Score> {
    surface_metric(y_true, y_pred, o, &|a, b| {
        (np_sum(a) + np_sum(b)) / (a.len() + b.len()) as f64
    })
}
