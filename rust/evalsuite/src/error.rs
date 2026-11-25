//! The error type shared by every metric.

use std::fmt;

/// Why a metric could not be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The inputs are malformed: wrong lengths, out-of-range values, unknown options.
    InvalidInput(String),
    /// The inputs are well-formed but the metric is mathematically undefined for them (for example ROC AUC
    /// with a single class present).
    Undefined(String),
    /// The metric does not support this kind of target (for example a binary-only metric on multiclass data).
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(m) | Error::Undefined(m) | Error::Unsupported(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::InvalidInput(msg.into()))
}

pub(crate) fn undefined<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Undefined(msg.into()))
}

pub(crate) fn unsupported<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(msg.into()))
}
