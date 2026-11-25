use evalsuite::generation as g;
use evalsuite::segmentation as s;
use std::time::Instant;

fn best<F: FnMut()>(mut f: F) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64() * 1e3);
    }
    b
}

fn main() {
    let words = [
        "the",
        "cat",
        "sat",
        "on",
        "a",
        "mat,",
        "dog",
        "ran",
        "home.",
        "3.5",
        "km",
        "(fast)",
        "it's",
        "well-known",
    ];
    let mut x: u64 = 7;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut sent = |n: usize| {
        (0..n)
            .map(|_| words[(rnd() % words.len() as u64) as usize])
            .collect::<Vec<_>>()
            .join(" ")
    };
    let preds: Vec<String> = (0..1000).map(|_| sent(20)).collect();
    let refs: Vec<Vec<String>> = (0..1000).map(|_| vec![sent(20)]).collect();
    println!(
        "bleu {:.2} ms",
        best(|| {
            g::bleu(&refs, &preds, &g::BleuOpts::default()).unwrap();
        })
    );
    println!(
        "chrf {:.2} ms",
        best(|| {
            g::chrf(&refs, &preds, &g::ChrfOpts::default()).unwrap();
        })
    );
    println!(
        "ter {:.2} ms",
        best(|| {
            g::ter(&refs, &preds, false).unwrap();
        })
    );
    let n = 512usize;
    let mk = |r: f64| {
        let labels: Vec<i64> = (0..n * n)
            .map(|i| {
                let (a, b) = ((i / n) as f64 - 256.0, (i % n) as f64 - 256.0);
                i64::from(a * a + b * b < r * r)
            })
            .collect();
        s::Mask::new(vec![n, n], labels).unwrap()
    };
    let (t, p) = (vec![mk(120.0)], vec![mk(118.0)]);
    let o = s::SurfaceOpts {
        num_classes: Some(2),
        macro_average: true,
        ..Default::default()
    };
    println!(
        "hd95 {:.2} ms",
        best(|| {
            s::hausdorff_distance(&t, &p, 95.0, &o).unwrap();
        })
    );
}
