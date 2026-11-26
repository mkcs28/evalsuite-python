//! C ABI: `include/evalsuite.h`. Every call takes a metric name and JSON arguments and returns a newly
//! allocated JSON string (free it with `evalsuite_free`); on error it returns NULL and
//! `evalsuite_last_error()` holds the message.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Computes `metric` on the JSON arguments `args_json`. Returns the JSON result (free with
/// `evalsuite_free`) or NULL on error.
///
/// # Safety
/// `metric` and `args_json` must be valid NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn evalsuite_call(
    metric: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    if metric.is_null() || args_json.is_null() {
        set_error("metric and args_json must not be NULL.".into());
        return std::ptr::null_mut();
    }
    // SAFETY: the caller guarantees valid NUL-terminated strings.
    let (m, a) = unsafe { (CStr::from_ptr(metric), CStr::from_ptr(args_json)) };
    let (Ok(m), Ok(a)) = (m.to_str(), a.to_str()) else {
        set_error("metric and args_json must be UTF-8.".into());
        return std::ptr::null_mut();
    };
    match evalsuite::bridge::call(m, a) {
        Ok(s) => CString::new(s).map_or(std::ptr::null_mut(), CString::into_raw),
        Err(e) => {
            set_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// The message of the last failed call on this thread (owned by the library; valid until the next call), or
/// NULL.
#[no_mangle]
pub extern "C" fn evalsuite_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
}

/// Frees a string returned by `evalsuite_call`.
///
/// # Safety
/// `s` must come from `evalsuite_call` (or be NULL) and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn evalsuite_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: `s` was produced by `CString::into_raw` in `evalsuite_call`.
        drop(unsafe { CString::from_raw(s) });
    }
}

/// The metric names `evalsuite_call` understands, as a JSON array (free with `evalsuite_free`).
#[no_mangle]
pub extern "C" fn evalsuite_metrics() -> *mut c_char {
    let names: Vec<String> = evalsuite::bridge::METRICS
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect();
    CString::new(format!("[{}]", names.join(","))).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// The engine version (static string).
#[no_mangle]
pub extern "C" fn evalsuite_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}
