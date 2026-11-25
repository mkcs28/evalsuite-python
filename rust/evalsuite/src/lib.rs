//! # EvalSuite
//!
//! Evaluation metrics and statistics for machine learning, NLP, computer vision, clinical prediction models
//! and LLM systems, in pure Rust. This is the Rust edition of the
//! [EvalSuite](https://pypi.org/project/evalsuite/) Python package: the same definitions, checked against the
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
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod agents;
pub mod calibration;
pub mod classification;
pub mod clinical;
pub mod detection;
pub mod error;
pub mod generation;
pub mod json_schema;
pub mod labels;
pub mod regression;
pub mod retrieval;
pub mod segmentation;
pub mod special;
pub mod structured;
pub mod summary;
pub mod stats;
pub mod uncertainty;
pub(crate) mod util;

pub use error::{Error, Result};
pub use serde_json::{json, Value};
pub use labels::Label;
