//! Object detection: box IoU and COCO-protocol average precision (exactly as pycocotools: greedy score-order
//! matching, crowd regions, area ranges, detection limits, 101-point interpolation), plus VOC-style AP. As
//! EvalSuite's Python `evalsuite.vision.detection`.

use crate::error::{invalid, Result};

/// Box coordinate convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoxFormat {
    /// (x1, y1, x2, y2).
    #[default]
    Xyxy,
    /// (x, y, width, height), as COCO.
    Xywh,
    /// (centre x, centre y, width, height), as YOLO.
    Cxcywh,
}

/// Ground truth and detections of one image.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Image {
    pub gt_boxes: Vec<[f64; 4]>,
    pub gt_labels: Vec<i64>,
    /// Crowd regions (default: none).
    pub gt_crowd: Option<Vec<bool>>,
    /// Areas used for the size ranges (default: box areas).
    pub gt_area: Option<Vec<f64>>,
    pub dt_boxes: Vec<[f64; 4]>,
    pub dt_labels: Vec<i64>,
    pub dt_scores: Vec<f64>,
}

fn to_xyxy(b: &[[f64; 4]], f: BoxFormat, name: &str) -> Result<Vec<[f64; 4]>> {
    let out: Vec<[f64; 4]> = b
        .iter()
        .map(|&[a, c, d, e]| match f {
            BoxFormat::Xyxy => [a, c, d, e],
            BoxFormat::Xywh => [a, c, a + d, c + e],
            BoxFormat::Cxcywh => [a - d / 2.0, c - e / 2.0, a + d / 2.0, c + e / 2.0],
        })
        .collect();
    if b.iter().flatten().any(|v| !v.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite coordinates."));
    }
    if out.iter().any(|x| x[2] < x[0] || x[3] < x[1]) {
        return invalid(format!(
            "{name} has boxes with negative width or height (check box_format)."
        ));
    }
    Ok(out)
}

fn area(b: &[f64; 4]) -> f64 {
    (b[2] - b[0]) * (b[3] - b[1])
}

/// Pairwise IoU matrix (rows: `a`, columns: `b`).
pub fn box_iou(a: &[[f64; 4]], b: &[[f64; 4]], format: BoxFormat) -> Result<Vec<Vec<f64>>> {
    let (a, b) = (
        to_xyxy(a, format, "boxes_a")?,
        to_xyxy(b, format, "boxes_b")?,
    );
    Ok(a.iter()
        .map(|d| {
            b.iter()
                .map(|g| {
                    let w = (d[2].min(g[2]) - d[0].max(g[0])).max(0.0);
                    let h = (d[3].min(g[3]) - d[1].max(g[1])).max(0.0);
                    let inter = w * h;
                    let union = area(d) + area(g) - inter;
                    if union > 0.0 {
                        inter / union
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect())
}

const AREAS: [(f64, f64); 4] = [(0.0, 1e10), (0.0, 1024.0), (1024.0, 9216.0), (9216.0, 1e10)];

fn coco_ious() -> Vec<f64> {
    crate::util::linspace(0.5, 0.95, 10)
}

struct Flat {
    gt_boxes: Vec<f64>,
    gt_labels: Vec<i64>,
    gt_crowd: Vec<bool>,
    gt_area: Vec<f64>,
    gt_off: Vec<usize>,
    dt_boxes: Vec<f64>,
    dt_labels: Vec<i64>,
    dt_scores: Vec<f64>,
    dt_off: Vec<usize>,
}

fn flatten(images: &[Image], f: BoxFormat) -> Result<Flat> {
    if images.is_empty() {
        return invalid("No images to evaluate.");
    }
    let mut fl = Flat {
        gt_boxes: vec![],
        gt_labels: vec![],
        gt_crowd: vec![],
        gt_area: vec![],
        gt_off: vec![0],
        dt_boxes: vec![],
        dt_labels: vec![],
        dt_scores: vec![],
        dt_off: vec![0],
    };
    for (i, im) in images.iter().enumerate() {
        let gb = to_xyxy(&im.gt_boxes, f, &format!("y_true[{i}]['boxes']"))?;
        let db = to_xyxy(&im.dt_boxes, f, &format!("y_pred[{i}]['boxes']"))?;
        if im.gt_labels.len() != gb.len() || im.dt_labels.len() != db.len() {
            return invalid(format!("Image {i}: labels must have one entry per box."));
        }
        if im.dt_scores.len() != db.len() || im.dt_scores.iter().any(|s| !s.is_finite()) {
            return invalid(format!(
                "y_pred[{i}]['scores'] must have one finite score per box."
            ));
        }
        let crowd = im.gt_crowd.clone().unwrap_or_else(|| vec![false; gb.len()]);
        let ar = im
            .gt_area
            .clone()
            .unwrap_or_else(|| gb.iter().map(area).collect());
        if crowd.len() != gb.len() || ar.len() != gb.len() {
            return invalid(format!(
                "y_true[{i}]: 'iscrowd' and 'area' need one value per box."
            ));
        }
        fl.gt_boxes.extend(gb.iter().flatten());
        fl.gt_labels.extend(&im.gt_labels);
        fl.gt_crowd.extend(crowd);
        fl.gt_area.extend(ar);
        fl.gt_off.push(fl.gt_labels.len());
        fl.dt_boxes.extend(db.iter().flatten());
        fl.dt_labels.extend(&im.dt_labels);
        fl.dt_scores.extend(&im.dt_scores);
        fl.dt_off.push(fl.dt_labels.len());
    }
    Ok(fl)
}

fn coco(
    images: &[Image],
    f: BoxFormat,
    thrs: &[f64],
    max_dets: &[usize],
    areas: &[(f64, f64)],
) -> Result<evalsuite_core::coco::Evaluation> {
    let fl = flatten(images, f)?;
    let img = evalsuite_core::coco::Images {
        gt_boxes: &fl.gt_boxes,
        gt_labels: &fl.gt_labels,
        gt_crowd: &fl.gt_crowd,
        gt_area: &fl.gt_area,
        gt_off: &fl.gt_off,
        dt_boxes: &fl.dt_boxes,
        dt_labels: &fl.dt_labels,
        dt_scores: &fl.dt_scores,
        dt_off: &fl.dt_off,
    };
    let rec = crate::util::linspace(0.0, 1.0, 101);
    Ok(evalsuite_core::coco::evaluate(
        &img, thrs, max_dets, areas, &rec,
    ))
}

fn mean_valid(v: impl Iterator<Item = f64>) -> f64 {
    let ok: Vec<f64> = v.filter(|&x| x > -1.0).collect();
    if ok.is_empty() {
        f64::NAN
    } else {
        crate::util::np_sum(&ok) / ok.len() as f64
    }
}

fn check_iou(t: f64) -> Result<f64> {
    if !(t > 0.0 && t < 1.0) {
        return invalid("iou_threshold must be between 0 and 1.");
    }
    Ok(t)
}

/// COCO mean average precision: over IoU 0.50:0.95 (`iou_threshold = None`) or at one threshold.
pub fn mean_average_precision(
    images: &[Image],
    iou_threshold: Option<f64>,
    format: BoxFormat,
    max_detections: usize,
) -> Result<f64> {
    let thrs = match iou_threshold {
        None => coco_ious(),
        Some(t) => vec![check_iou(t)?],
    };
    let e = coco(images, format, &thrs, &[max_detections], &AREAS[..1])?;
    Ok(mean_valid(e.precision.iter().copied()))
}

/// The twelve COCO summary numbers (mAP, mAP50, mAP75, small/medium/large, mAR@1/10/100, small/medium/large).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionReport {
    pub values: Vec<(&'static str, f64)>,
    /// Per-class mAP@[.50:.95] (all areas).
    pub per_class: Vec<(i64, f64)>,
}

/// Full COCO evaluation summary.
pub fn detection_report(
    images: &[Image],
    format: BoxFormat,
    max_detections: usize,
) -> Result<DetectionReport> {
    if max_detections < 1 {
        return invalid("max_detections must be a positive integer.");
    }
    let mut md = vec![1, 10, max_detections];
    md.sort_unstable();
    md.dedup();
    let e = coco(images, format, &coco_ious(), &md, &AREAS)?;
    let (t_n, r_n, k_n, a_n, m_n) = (10, 101, e.cats.len(), 4, md.len());
    let m = md.iter().position(|&x| x == max_detections).unwrap();
    let p = |t: usize, r: usize, k: usize, a: usize, mi: usize| {
        e.precision[(((t * r_n + r) * k_n + k) * a_n + a) * m_n + mi]
    };
    let rc =
        |t: usize, k: usize, a: usize, mi: usize| e.recall[((t * k_n + k) * a_n + a) * m_n + mi];
    let ap = |t: Option<usize>, a: usize| {
        let ts: Vec<usize> = t.map_or_else(|| (0..t_n).collect(), |t| vec![t]);
        mean_valid(
            ts.iter()
                .flat_map(|&t| (0..r_n).flat_map(move |r| (0..k_n).map(move |k| p(t, r, k, a, m)))),
        )
    };
    let ar = |a: usize, mi: usize| {
        mean_valid((0..t_n).flat_map(|t| (0..k_n).map(move |k| rc(t, k, a, mi))))
    };
    let mi = |d: usize| md.iter().position(|&x| x == d).unwrap();
    let values = vec![
        ("map", ap(None, 0)),
        ("map_50", ap(Some(0), 0)),
        ("map_75", ap(Some(5), 0)),
        ("map_small", ap(None, 1)),
        ("map_medium", ap(None, 2)),
        ("map_large", ap(None, 3)),
        ("mar_1", ar(0, mi(1))),
        ("mar_10", ar(0, mi(10))),
        (
            if max_detections == 100 {
                "mar_100"
            } else {
                "mar_max"
            },
            ar(0, m),
        ),
        ("mar_small", ar(1, m)),
        ("mar_medium", ar(2, m)),
        ("mar_large", ar(3, m)),
    ];
    let per_class = e
        .cats
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            (
                c,
                mean_valid((0..t_n).flat_map(|t| (0..r_n).map(move |r| p(t, r, k, 0, m)))),
            )
        })
        .collect();
    Ok(DetectionReport { values, per_class })
}

/// Precision interpolation for [`average_precision_detection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// 101 recall points, as pycocotools.
    #[default]
    Coco,
    /// Every recall change (PASCAL VOC 2010+).
    Voc,
}

/// Per-class average precision at one IoU threshold: `(classes, AP)` (NaN for classes without ground truth).
pub fn average_precision_detection(
    images: &[Image],
    iou_threshold: f64,
    interpolation: Interpolation,
    format: BoxFormat,
    max_detections: usize,
) -> Result<(Vec<i64>, Vec<f64>)> {
    let t = check_iou(iou_threshold)?;
    match interpolation {
        Interpolation::Coco => {
            let e = coco(images, format, &[t], &[max_detections], &AREAS[..1])?;
            let k_n = e.cats.len();
            let per = (0..k_n)
                .map(|k| mean_valid((0..101).map(|r| e.precision[r * k_n + k])))
                .collect();
            Ok((e.cats, per))
        }
        Interpolation::Voc => voc(images, t, format, max_detections),
    }
}

fn stable_desc(s: &[f64]) -> Vec<usize> {
    evalsuite_core::curve::descending_order(s)
}

fn voc(
    images: &[Image],
    t: f64,
    format: BoxFormat,
    max_det: usize,
) -> Result<(Vec<i64>, Vec<f64>)> {
    let fl = flatten(images, format)?;
    let mut cats: Vec<i64> = fl.gt_labels.iter().chain(&fl.dt_labels).copied().collect();
    cats.sort_unstable();
    cats.dedup();
    let mut per = vec![f64::NAN; cats.len()];
    for (k, &cat) in cats.iter().enumerate() {
        // Per image: score-sorted detections (truncated), ground truth with ignored (crowd) last, IoUs.
        let (mut scores, mut matched, mut ignored) = (Vec::new(), Vec::new(), Vec::new());
        let mut npig = 0usize;
        for i in 0..images.len() {
            let g: Vec<usize> = (fl.gt_off[i]..fl.gt_off[i + 1])
                .filter(|&j| fl.gt_labels[j] == cat)
                .collect();
            let d: Vec<usize> = (fl.dt_off[i]..fl.dt_off[i + 1])
                .filter(|&j| fl.dt_labels[j] == cat)
                .collect();
            let ds: Vec<f64> = d.iter().map(|&j| fl.dt_scores[j]).collect();
            let order: Vec<usize> = stable_desc(&ds).into_iter().take(max_det).collect();
            let d: Vec<usize> = order.iter().map(|&o| d[o]).collect();
            let gign: Vec<bool> = g.iter().map(|&j| fl.gt_crowd[j]).collect();
            let mut gorder: Vec<usize> = (0..g.len()).collect();
            gorder.sort_by_key(|&x| gign[x]);
            npig += gign.iter().filter(|x| !**x).count();
            let (gn, dn) = (g.len(), d.len());
            if dn == 0 {
                continue;
            }
            let bx = |v: &[f64], j: usize| [v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]];
            let mut ious = vec![-1.0f64; dn * gn.max(1)];
            for (a, &dj) in d.iter().enumerate() {
                let db = bx(&fl.dt_boxes, dj);
                for (b, &go) in gorder.iter().enumerate() {
                    let gj = g[go];
                    let gb = bx(&fl.gt_boxes, gj);
                    let w = (db[2].min(gb[2]) - db[0].max(gb[0])).max(0.0);
                    let h = (db[3].min(gb[3]) - db[1].max(gb[1])).max(0.0);
                    let inter = w * h;
                    let union = if fl.gt_crowd[gj] {
                        area(&db)
                    } else {
                        area(&db) + area(&gb) - inter
                    };
                    ious[a * gn.max(1) + b] = if union > 0.0 { inter / union } else { 0.0 };
                }
            }
            let gi: Vec<bool> = if gn == 0 {
                vec![true]
            } else {
                gorder.iter().map(|&x| gign[x]).collect()
            };
            let cr: Vec<bool> = if gn == 0 {
                vec![false]
            } else {
                gorder.iter().map(|&x| fl.gt_crowd[g[x]]).collect()
            };
            let out = vec![false; dn];
            let (m, ig) =
                evalsuite_core::coco::match_images(&ious, &gi, &cr, &out, 1, dn, gn.max(1), &[t]);
            for (a, &dj) in d.iter().enumerate() {
                scores.push(fl.dt_scores[dj]);
                matched.push(m[a]);
                ignored.push(ig[a]);
            }
        }
        if npig == 0 {
            continue;
        }
        let order = stable_desc(&scores);
        let (mut tp, mut fp) = (Vec::new(), Vec::new());
        let (mut ctp, mut cfp) = (0.0f64, 0.0f64);
        for &o in &order {
            ctp += f64::from(u8::from(matched[o] && !ignored[o]));
            cfp += f64::from(u8::from(!matched[o] && !ignored[o]));
            if !ignored[o] {
                tp.push(ctp);
                fp.push(cfp);
            }
        }
        if tp.is_empty() {
            per[k] = 0.0;
            continue;
        }
        let rc: Vec<f64> = tp.iter().map(|x| x / npig as f64).collect();
        let pr: Vec<f64> = tp
            .iter()
            .zip(&fp)
            .map(|(a, b)| a / (a + b).max(f64::EPSILON))
            .collect();
        let mut mrec = vec![0.0];
        mrec.extend(&rc);
        mrec.push(1.0);
        let mut mpre = vec![0.0];
        mpre.extend(&pr);
        mpre.push(0.0);
        for i in (0..mpre.len() - 1).rev() {
            mpre[i] = mpre[i].max(mpre[i + 1]);
        }
        let terms: Vec<f64> = (0..mrec.len() - 1)
            .filter(|&i| mrec[i + 1] != mrec[i])
            .map(|i| (mrec[i + 1] - mrec[i]) * mpre[i + 1])
            .collect();
        per[k] = crate::util::np_sum(&terms);
    }
    Ok((cats, per))
}
