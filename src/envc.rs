//! The process environment, read through a cache.
//!
//! evoview is configured by ~550 `EVOVIEW_*` lookups, many of them inside
//! the frame loop (per car, per pass, per uniform). On Windows each
//! `std::env::var` takes the environment lock, asks the OS and converts
//! UTF-16, and a sampled 53-car replay frame spent ~9% of its CPU there.
//! Values are cached per key on first read.
//!
//! ⚠ Every change to the environment must go through `set_var` /
//! `remove_var` here, which drop the key from the cache - evoview sets a
//! few variables itself (a look while loading an AI car, the render scale
//! DLSS / FSR pick, the driving defaults). Iterating `std::env::vars()` is
//! not cached.

use std::collections::HashMap;
use std::env::VarError;
use std::ffi::{OsStr, OsString};
use std::sync::RwLock;

static CACHE: RwLock<Option<HashMap<OsString, Option<OsString>>>> = RwLock::new(None);

/// `std::env::var_os`, cached.
pub fn var_os<K: AsRef<OsStr>>(key: K) -> Option<OsString> {
    let k = key.as_ref();
    if let Ok(c) = CACHE.read() {
        if let Some(v) = c.as_ref().and_then(|m| m.get(k)) {
            return v.clone();
        }
    }
    let v = std::env::var_os(k);
    if let Ok(mut c) = CACHE.write() {
        c.get_or_insert_with(HashMap::new).insert(k.to_os_string(), v.clone());
    }
    v
}

/// `std::env::var`, cached (the same errors).
pub fn var<K: AsRef<OsStr>>(key: K) -> Result<String, VarError> {
    match var_os(key) {
        Some(s) => s.into_string().map_err(VarError::NotUnicode),
        None => Err(VarError::NotPresent),
    }
}

fn forget(k: &OsStr) {
    if let Ok(mut c) = CACHE.write() {
        if let Some(m) = c.as_mut() {
            m.remove(k);
        }
    }
}

/// `std::env::set_var`, keeping the cache true.
pub fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    std::env::set_var(key.as_ref(), value);
    forget(key.as_ref());
}

/// `std::env::remove_var`, keeping the cache true.
pub fn remove_var<K: AsRef<OsStr>>(key: K) {
    std::env::remove_var(key.as_ref());
    forget(key.as_ref());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_set_and_remove() {
        let k = "EVOVIEW_ENVC_TEST_KEY";
        remove_var(k);
        assert!(var(k).is_err());
        set_var(k, "1.5");
        assert_eq!(var(k).as_deref(), Ok("1.5"));
        set_var(k, "2");
        assert_eq!(var(k).as_deref(), Ok("2"));
        assert_eq!(var_os(k), Some(OsString::from("2")));
        remove_var(k);
        assert_eq!(var(k), Err(VarError::NotPresent));
        assert_eq!(var_os(k), None);
    }
}
