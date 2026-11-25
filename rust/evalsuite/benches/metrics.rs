//! `cargo bench -p evalsuite`: timings of representative metrics at realistic sizes (see BENCHMARKS.md).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use evalsuite::classification::{self as cls, Average, Opts};
use evalsuite::{generation, regression, retrieval, segmentation, stats};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn classification(c: &mut Criterion) {
    let mut g = c.benchmark_group("classification");
    g.sample_size(20);
    for n in [100_000usize, 1_000_000] {
        let mut r = Rng(1);
        let y: Vec<i64> = (0..n).map(|_| (r.next() % 2) as i64).collect();
        let p: Vec<i64> = (0..n).map(|_| (r.next() % 2) as i64).collect();
        let prob: Vec<f64> = (0..n).map(|_| r.unit()).collect();
        let y5: Vec<i64> = (0..n).map(|_| (r.next() % 5) as i64).collect();
        let p5: Vec<i64> = (0..n).map(|_| (r.next() % 5) as i64).collect();
        let o = Opts {
            average: Average::Macro,
            ..Opts::default()
        };
        g.bench_with_input(BenchmarkId::new("accuracy", n), &n, |b, _| {
            b.iter(|| cls::accuracy(black_box(&y), black_box(&p), None))
        });
        g.bench_with_input(BenchmarkId::new("f1_macro_5class", n), &n, |b, _| {
            b.iter(|| cls::f1(black_box(&y5), black_box(&p5), &o))
        });
        g.bench_with_input(BenchmarkId::new("roc_auc", n), &n, |b, _| {
            b.iter(|| cls::binary::roc_auc(black_box(&y), black_box(&prob), None, None))
        });
    }
    g.finish();
}

fn regression_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("regression");
    g.sample_size(20);
    let n = 1_000_000;
    let mut r = Rng(2);
    let y: Vec<f64> = (0..n).map(|_| r.unit()).collect();
    let p: Vec<f64> = (0..n).map(|_| r.unit()).collect();
    g.bench_function("rmse/1000000", |b| {
        b.iter(|| regression::rmse(black_box(&y), black_box(&p), None))
    });
    g.bench_function("r2/1000000", |b| {
        b.iter(|| regression::r2(black_box(&y), black_box(&p), None))
    });
    g.bench_function("median_absolute_error/1000000", |b| {
        b.iter(|| regression::median_absolute_error(black_box(&y), black_box(&p), None))
    });
    g.finish();
}

fn statistics(c: &mut Criterion) {
    let mut g = c.benchmark_group("stats");
    g.sample_size(10);
    let mut r = Rng(3);
    let x: Vec<f64> = (0..10_000).map(|_| r.unit()).collect();
    let y: Vec<f64> = (0..10_000).map(|_| r.unit() + 0.01).collect();
    // Bootstrap (BCa, the default) of the MAE between x and y.
    let mae = |idx: &[usize]| {
        Ok(idx.iter().map(|&i| (x[i] - y[i]).abs()).sum::<f64>() / idx.len() as f64)
    };
    let opts = stats::Bootstrap {
        n_resamples: 2000,
        seed: 0,
        ..Default::default()
    };
    g.bench_function("bootstrap_ci_mae/10000x2000", |b| {
        b.iter(|| stats::bootstrap_ci(x.len(), mae, &opts))
    });
    g.bench_function("mann_whitney/10000", |b| {
        b.iter(|| {
            stats::mann_whitney_test(black_box(&x), black_box(&y), stats::Alternative::TwoSided)
        })
    });
    g.bench_function("t_test/10000", |b| {
        b.iter(|| {
            stats::t_test(
                black_box(&x),
                black_box(&y),
                false,
                stats::Alternative::TwoSided,
                0.95,
            )
        })
    });
    g.finish();
}

fn text(c: &mut Criterion) {
    let mut g = c.benchmark_group("text");
    g.sample_size(10);
    let words = [
        "the", "cat", "sat", "on", "a", "mat", "dog", "ran", "fast", "slowly", "and", "then", "it",
        "slept",
    ];
    let mut r = Rng(4);
    let mut sent = |len: usize| {
        (0..len)
            .map(|_| words[(r.next() % words.len() as u64) as usize])
            .collect::<Vec<_>>()
            .join(" ")
    };
    let preds: Vec<String> = (0..1000).map(|_| sent(20)).collect();
    let refs: Vec<Vec<String>> = (0..1000).map(|_| vec![sent(20)]).collect();
    g.bench_function("bleu/1000", |b| {
        b.iter(|| {
            generation::bleu(
                black_box(&refs),
                black_box(&preds),
                &generation::BleuOpts::default(),
            )
        })
    });
    g.bench_function("chrf/1000", |b| {
        b.iter(|| {
            generation::chrf(
                black_box(&refs),
                black_box(&preds),
                &generation::ChrfOpts::default(),
            )
        })
    });
    g.bench_function("ter/1000", |b| {
        b.iter(|| generation::ter(black_box(&refs), black_box(&preds), false))
    });
    g.bench_function("rouge_l/1000", |b| {
        b.iter(|| generation::rouge_l(black_box(&refs), black_box(&preds)))
    });
    g.finish();
}

fn retrieval_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("retrieval");
    g.sample_size(10);
    let mut r = Rng(5);
    let rel: Vec<_> = (0..10_000)
        .map(|_| retrieval::judgements((0..5).map(|_| r.next() % 100)))
        .collect();
    let run: Vec<Vec<u64>> = (0..10_000)
        .map(|_| {
            let mut v: Vec<u64> = (0..100).collect();
            for i in (1..v.len()).rev() {
                v.swap(i, (r.next() % (i as u64 + 1)) as usize);
            }
            v.truncate(20);
            v
        })
        .collect();
    g.bench_function("ndcg_at_10/10000_queries", |b| {
        b.iter(|| {
            retrieval::ndcg_at_k(
                black_box(&rel),
                black_box(&run),
                Some(10),
                retrieval::Gains::Linear,
            )
        })
    });
    g.bench_function("mrr/10000_queries", |b| {
        b.iter(|| retrieval::mrr(black_box(&rel), black_box(&run), None))
    });
    g.finish();
}

fn segmentation_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("segmentation");
    g.sample_size(10);
    let (h, w) = (512usize, 512usize);
    let blob = |cx: f64, cy: f64| -> Vec<i64> {
        (0..h * w)
            .map(|i| {
                let (y, x) = ((i / w) as f64, (i % w) as f64);
                i64::from(((x - cx).powi(2) + (y - cy).powi(2)).sqrt() < 120.0)
            })
            .collect()
    };
    let t = segmentation::Mask {
        shape: vec![h, w],
        labels: blob(256.0, 256.0),
    };
    let p = segmentation::Mask {
        shape: vec![h, w],
        labels: blob(270.0, 250.0),
    };
    let o = segmentation::Opts::default();
    g.bench_function("dice/512x512", |b| {
        b.iter(|| {
            segmentation::dice(
                black_box(std::slice::from_ref(&t)),
                black_box(std::slice::from_ref(&p)),
                &o,
            )
        })
    });
    g.bench_function("hausdorff_95/512x512", |b| {
        b.iter(|| {
            segmentation::hausdorff_distance(
                black_box(std::slice::from_ref(&t)),
                black_box(std::slice::from_ref(&p)),
                95.0,
                &segmentation::SurfaceOpts::default(),
            )
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    classification,
    regression_bench,
    statistics,
    text,
    retrieval_bench,
    segmentation_bench
);
criterion_main!(benches);
