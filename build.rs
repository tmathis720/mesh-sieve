// build.rs

//! build.rs — locate METIS, generate Rust FFI bindings, and post‐process them
//!
//! This script supports two discovery modes:
//!  • Default (pkg-config):  use `pkg_config::probe("metis")`
//!  • Manual    (env-vars):  `METIS_NO_PKG_CONFIG=1` plus `METIS_DIR` or
//!                           (`METIS_LIB_DIR` + `METIS_INCLUDE_DIR`).
//!
//! After generating `metis_bindings.rs`, we post-process it to replace
//! every `extern "C" { … }` with `unsafe extern "C" { … }`, which Rust now requires.
//! We also strip out any `unsafe impl Send/Sync for idx_t` lines so you won’t
//! get orphan-rule errors if you have them lingering in the generated file.

#[cfg(feature = "metis-support")]
fn main() {
    use regex::Regex;
    use std::env;
    use std::fs::{self, read_to_string, write};
    use std::path::Path;

    // ─── 1. Find METIS include/lib directories ───────────────────────────────────
    let (include_dir, lib_dir) = if env::var_os("METIS_NO_PKG_CONFIG").is_some() {
        // -------- Manual (Option B) ----------
        let prefix =
            env::var("METIS_DIR").expect("METIS_DIR must be set when METIS_NO_PKG_CONFIG=1");

        let inc = env::var("METIS_INCLUDE_DIR").unwrap_or_else(|_| format!("{}/include", &prefix));

        let lib = env::var("METIS_LIB_DIR").unwrap_or_else(|_| format!("{}/lib", &prefix));

        // Tell Cargo to look in `$lib` for both libmetis.so and libGKlib.a:
        println!("cargo:rustc-link-search=native={}", lib);
        println!("cargo:rustc-link-lib=dylib=metis");
        println!("cargo:rustc-link-lib=dylib=GKlib");

        println!("cargo:rustc-link-search=native=/opt/intel/oneapi/compiler/2025.1/lib");
        println!("cargo:rustc-link-lib=dylib=intlc");

        println!("cargo:rerun-if-env-changed=METIS_DIR");
        println!("cargo:rerun-if-env-changed=METIS_LIB_DIR");
        println!("cargo:rerun-if-env-changed=METIS_INCLUDE_DIR");

        (inc, lib)
    } else {
        // -------- pkg-config (Option A) ----------
        let lib = pkg_config::Config::new()
            .statik(false)
            .probe("metis")
            .expect("Could not find METIS via pkg-config; set METIS_NO_PKG_CONFIG=1 to bypass");

        // pkg-config already printed:
        //   cargo:rustc-link-search=native=/path/to/metis/lib
        //   cargo:rustc-link-lib=dylib=metis
        // All we need to add is GKlib:
        println!("cargo:rustc-link-lib=dylib=GKlib");

        let inc = lib
            .include_paths
            .get(0)
            .unwrap_or_else(|| panic!("pkg-config returned no include path for METIS"))
            .display()
            .to_string();

        let lib_dir = lib
            .link_paths
            .get(0)
            .unwrap_or_else(|| panic!("pkg-config returned no library path for METIS"))
            .display()
            .to_string();

        (inc, lib_dir)
    };

    // ─── 2. Generate + post-process `metis_bindings.rs` ─────────────────────────
    // (Remains exactly as before: bindgen → patch “extern” → write → delete raw)

    let out_path = Path::new("src").join("metis_bindings_raw.rs");
    let final_path = Path::new("src").join("metis_bindings.rs");

    let bindings = bindgen::Builder::default()
        .header(format!("{}/metis.h", include_dir))
        .allowlist_function("METIS_.*")
        .allowlist_type("idx_t")
        .generate()
        .expect("Failed to generate METIS bindings via bindgen");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write raw bindings to src/metis_bindings_raw.rs");

    // Post‐process raw bindings: convert every `extern "C"` → `unsafe extern "C"`,
    // and remove any `unsafe impl Send/Sync for idx_t { }` lines if they exist.
    let raw_contents = read_to_string(&out_path)
        .expect("Unable to read src/metis_bindings_raw.rs for post-processing");

    let re_extern = Regex::new(r#"(?m)^(?P<prefix>\s*)(?P<block>extern\s+"C"\s*\{)"#)
        .expect("Invalid regex for extern block");
    let re_strip_send_sync =
        Regex::new(r#"(?m)^\s*unsafe\s+impl\s+(Send|Sync)\s+for\s+idx_t\s*\{\s*\}\s*$"#)
            .expect("Invalid regex for stripping Send/Sync impls");

    let with_unsafe_extern = re_extern.replace_all(&raw_contents, |caps: &regex::Captures| {
        format!("{}unsafe {}", &caps["prefix"], &caps["block"])
    });
    let cleaned: String = re_strip_send_sync
        .replace_all(&with_unsafe_extern, "")
        .to_string();

    write(&final_path, cleaned)
        .expect("Unable to write post-processed bindings to src/metis_bindings.rs");
    fs::remove_file(&out_path).expect("Unable to remove temporary metis_bindings_raw.rs");

    // ─── 3. Re-run triggers ──────────────────────────────────────────────────────
    println!("cargo:rerun-if-env-changed=METIS_NO_PKG_CONFIG");
    println!("cargo:rerun-if-env-changed=METIS_DIR");
    println!("cargo:rerun-if-env-changed=METIS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=METIS_INCLUDE_DIR");
    println!("cargo:rerun-if-changed={}/metis.h", include_dir);
}

#[cfg(not(feature = "metis-support"))]
fn main() {
    // No-op when the “metis-support” feature is disabled
}
