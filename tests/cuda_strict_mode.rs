#[test]
fn explicitly_requested_cuda_tests_require_the_cuda_feature() {
    if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1")
        && !cfg!(feature = "cuda")
    {
        panic!("MESH_SIEVE_RUN_CUDA_TESTS=1 requires building tests with `--features cuda`");
    }
}
