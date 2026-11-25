//! # EvalSuite
//!
//! Evaluation metrics and statistics for machine learning, NLP, computer vision, clinical prediction models
//! and LLM systems, in pure Rust. This is the Rust edition of the
//! [EvalSuite](https://pypi.org/project/evalsuite-python/) Python package: the same definitions, checked against the
//! Python implementation (which itself is checked against scikit-learn, SciPy, statsmodels, sacreBLEU,
//! pycocotools and the other reference libraries).
//!
//! Every function returns [`Result`]: malformed inputs give [`Error::InvalidInput`], mathematically undefined
//! cases (ROC AUC with one class, …) give [`Error::Undefined`].
//!
//! ```
//! use evalsuite::{classification, regression};
//! let acc = classification::accuracy(&[1, 0, 1, 1], &[1, 0, 0, 1], None).unwrap();
//! assert_eq!(acc, 0.75);
//! let rmse = regression::rmse(&[3.0, -0.5, 2.0, 7.0], &[2.5, 0.0, 2.0, 8.0], None).unwrap();
//! assert!((rmse - 0.6123724356957945).abs() < 1e-15);
//! ```
#![cfg_attr(doctest, doc = include_str!("../README.md"))]
#![forbid(unsafe_code)]
// Reference constants are kept with the digits of their published sources; index loops mirror the
// reference formulas.
#![allow(
    clippy::excessive_precision,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord
)]
#![warn(missing_debug_implementations)]

pub mod agents;
pub mod calibration;
pub mod classification;
pub mod clinical;
pub mod code;
pub mod detection;
pub mod efficiency;
pub mod error;
pub mod factuality;
pub mod generation;
pub mod json_schema;
pub mod judge;
pub mod labels;
pub mod longcontext;
pub mod monitor;
pub mod multilingual;
#[cfg(feature = "python")]
pub(crate) mod pyast;
pub mod qa;
pub mod rag;
pub mod regression;
pub mod retrieval;
pub mod robustness;
pub mod safety;
pub mod segmentation;
pub mod semantic;
pub mod special;
pub mod spice;
pub mod stats;
pub mod structured;
pub mod summary;
pub mod uncertainty;
pub(crate) mod util;

pub use error::{Error, Result};
pub use labels::Label;
pub use serde_json::{json, Value};
