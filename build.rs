fn main() {
    if cfg!(unix) {
        if let Some(lib_dir) = &pyo3_build_config::get().lib_dir {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
        }
    }
}
