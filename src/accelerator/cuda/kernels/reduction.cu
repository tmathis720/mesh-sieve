// Deterministic hierarchical reductions. Each block emits one stable partial;
// the executor reduces those partials in a second launch.
#define DEFINE_REDUCTIONS(TYPE, SUFFIX)                                             \
extern "C" __global__ void sum_##SUFFIX(const TYPE* x, TYPE* out, unsigned int n) {\
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) local += (double)x[i];\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] += values[threadIdx.x + s];         \
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[blockIdx.x] = (TYPE)values[0];                        \
}                                                                                   \
extern "C" __global__ void dot_##SUFFIX(const TYPE* a, const TYPE* b, TYPE* out, unsigned int n) {\
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) local += (double)a[i] * (double)b[i];\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] += values[threadIdx.x + s];         \
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[blockIdx.x] = (TYPE)values[0];                        \
}                                                                                   \
extern "C" __global__ void l2_##SUFFIX(const TYPE* x, TYPE* out, unsigned int n) { \
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) { double v=(double)x[i]; local += v*v; }\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] += values[threadIdx.x + s];         \
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[blockIdx.x] = (TYPE)values[0];                        \
}                                                                                   \
extern "C" __global__ void max_abs_##SUFFIX(const TYPE* x, TYPE* out, unsigned int n) {\
    __shared__ double values[256];                                                   \
    double local = 0.0;                                                             \
    for (unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) local = fmax(local, fabs((double)x[i]));\
    values[threadIdx.x] = local; __syncthreads();                                   \
    for (unsigned int s = blockDim.x / 2; s; s >>= 1) {                             \
        if (threadIdx.x < s) values[threadIdx.x] = fmax(values[threadIdx.x], values[threadIdx.x + s]);\
        __syncthreads();                                                            \
    }                                                                               \
    if (threadIdx.x == 0) out[blockIdx.x] = (TYPE)values[0];                        \
}                                                                                   \
extern "C" __global__ void sqrt_##SUFFIX(TYPE* value) {                            \
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] = (TYPE)sqrt((double)value[0]);\
}

DEFINE_REDUCTIONS(float, f32)
DEFINE_REDUCTIONS(double, f64)
