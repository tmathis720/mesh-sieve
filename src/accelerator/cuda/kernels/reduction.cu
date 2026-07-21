// Deterministic one-block reductions. Each thread traverses a fixed strided
// subsequence and the shared-memory tree has a stable order.
#define DEFINE_REDUCTIONS(TYPE, SUFFIX)                                             \
extern "C" __global__ void sum_##SUFFIX(const TYPE* x, TYPE* out, unsigned int n) {\
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) local += (double)x[i];\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] += values[threadIdx.x + s];         \
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[0] = (TYPE)values[0];                                 \
}                                                                                   \
extern "C" __global__ void dot_##SUFFIX(const TYPE* a, const TYPE* b, TYPE* out, unsigned int n) {\
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) local += (double)a[i] * (double)b[i];\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] += values[threadIdx.x + s];         \
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[0] = (TYPE)values[0];                                 \
}                                                                                   \
extern "C" __global__ void l2_##SUFFIX(const TYPE* x, TYPE* out, unsigned int n) { \
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) { double v=(double)x[i]; local += v*v; }\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] += values[threadIdx.x + s];         \
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[0] = (TYPE)sqrt(values[0]);                           \
}                                                                                   \
extern "C" __global__ void max_abs_##SUFFIX(const TYPE* x, TYPE* out, unsigned int n) {\
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) local = fmax(local, fabs((double)x[i]));\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] = fmax(values[threadIdx.x], values[threadIdx.x + s]);\
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[0] = (TYPE)values[0];                                 \
}

DEFINE_REDUCTIONS(float, f32)
DEFINE_REDUCTIONS(double, f64)
