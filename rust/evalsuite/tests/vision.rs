mod common;

use common::*;
use evalsuite::classification::Score;
use evalsuite::detection::{self as d, BoxFormat, Image, Interpolation};
use evalsuite::segmentation::{self as s, Aggregate, Average, Mask, Opts, SurfaceOpts};
use serde_json::Value;

fn masks(v: &Value, shape: &[usize]) -> Vec<Mask> {
    fn flat(v: &Value, out: &mut Vec<i64>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| flat(x, out)),
            x => out.push(x.as_i64().unwrap()),
        }
    }
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let mut l = Vec::new();
            flat(m, &mut l);
            Mask::new(shape.to_vec(), l).unwrap()
        })
        .collect()
}

fn sc(r: evalsuite::Result<Score>) -> Got {
    match r {
        Ok(Score::Scalar(v)) => Got::Num(v),
        Ok(Score::PerClass(v)) => Got::List(v),
        Err(e) => Got::Err(e.to_string()),
    }
}

#[test]
fn segmentation_matches_python() {
    let (n, sk) = check("segmentation", 1e-10, |f, a| {
        let shape: Vec<usize> = ints(&a["shape"]).into_iter().map(|x| x as usize).collect();
        let (t, p) = (masks(&a["y_true"], &shape), masks(&a["y_pred"], &shape));
        let ign = a["ignore_index"].as_i64();
        let k = a["num_classes"].as_u64().map(|k| k as usize);
        let avg = match a["average"].as_str() {
            Some("micro") => Average::Micro,
            Some("weighted") => Average::Weighted,
            Some("none") => Average::None,
            _ => Average::Macro,
        };
        let o = Opts {
            num_classes: k,
            ignore_index: ign,
            average: avg,
            aggregate: if a["aggregate"] == "image" {
                Aggregate::Image
            } else {
                Aggregate::Dataset
            },
            empty: a["empty"].as_f64(),
            ..Default::default()
        };
        let so = SurfaceOpts {
            spacing: opt_nums(&a["spacing"]),
            num_classes: k,
            ignore_index: ign,
            macro_average: avg == Average::Macro,
            ..Default::default()
        };
        let bo = Opts {
            include_background: false,
            ..o.clone()
        };
        match f {
            "dice" => sc(s::dice(&t, &p, &o)),
            "iou" => sc(s::iou(&t, &p, &o)),
            "pixel_accuracy" => {
                s::pixel_accuracy(&t, &p, ign).map_or_else(|e| Got::Err(e.to_string()), Got::Num)
            }
            "mean_pixel_accuracy" => s::mean_pixel_accuracy(&t, &p, k, ign)
                .map_or_else(|e| Got::Err(e.to_string()), Got::Num),
            "boundary_iou" => sc(s::boundary_iou(&t, &p, 0.05, &bo)),
            "hausdorff_distance" => sc(s::hausdorff_distance(&t, &p, num(&a["percentile"]), &so)),
            "average_surface_distance" => sc(s::average_surface_distance(&t, &p, &so)),
            _ => Got::Skip,
        }
    });
    assert!(n > 900 && sk == 0);
}

fn images(a: &Value) -> Vec<Image> {
    let boxes = |v: &Value| -> Vec<[f64; 4]> {
        rows(v)
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect()
    };
    a["y_true"]
        .as_array()
        .unwrap()
        .iter()
        .zip(a["y_pred"].as_array().unwrap())
        .map(|(g, p)| Image {
            gt_boxes: boxes(&g["boxes"]),
            gt_labels: ints(&g["labels"]),
            gt_crowd: Some(
                g["iscrowd"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_bool().unwrap())
                    .collect(),
            ),
            gt_area: None,
            dt_boxes: boxes(&p["boxes"]),
            dt_labels: ints(&p["labels"]),
            dt_scores: nums(&p["scores"]),
        })
        .collect()
}

#[test]
fn detection_matches_python() {
    let (n, sk) = check("detection", 1e-12, |f, a| match f {
        "box_iou" => {
            let b = |v: &Value| -> Vec<[f64; 4]> {
                rows(v)
                    .into_iter()
                    .map(|r| [r[0], r[1], r[2], r[3]])
                    .collect()
            };
            d::box_iou(&b(&a["a"]), &b(&a["b"]), BoxFormat::Xyxy)
                .map_or_else(|e| Got::Err(e.to_string()), |m| Got::List(m.concat()))
        }
        "mean_average_precision" => d::mean_average_precision(
            &images(a),
            a["iou_threshold"].as_f64(),
            BoxFormat::Xyxy,
            100,
        )
        .map_or_else(|e| Got::Err(e.to_string()), Got::Num),
        "average_precision_detection" => {
            let interp = if a["interpolation"] == "voc" {
                Interpolation::Voc
            } else {
                Interpolation::Coco
            };
            d::average_precision_detection(&images(a), 0.5, interp, BoxFormat::Xyxy, 100)
                .map_or_else(|e| Got::Err(e.to_string()), |(_, v)| Got::List(v))
        }
        "detection_report" => d::detection_report(&images(a), BoxFormat::Xyxy, 100).map_or_else(
            |e| Got::Err(e.to_string()),
            |r| Got::List(r.values.iter().map(|x| x.1).collect()),
        ),
        _ => Got::Skip,
    });
    assert!(n > 100 && sk == 0);
}
