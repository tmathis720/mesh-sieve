// Scalar conservative FVM kernels. Face fluxes are written once, then each
// cell gathers its incident contributions in deterministic CSR order.

#define DEFINE_FVM_KERNELS(TYPE, SUFFIX)                                            \
extern "C" __global__ void internal_flux_##SUFFIX(                                  \
    const unsigned int* owner, const unsigned int* neighbor,                        \
    const unsigned char* active, const TYPE* cell, const TYPE* mass,                \
    TYPE* flux, unsigned int scheme, unsigned int count) {                          \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i >= count) return;                                                         \
    if (!active[i]) { flux[i] = (TYPE)0; return; }                                  \
    const TYPE m = mass[i];                                                         \
    const TYPE l = cell[owner[i]];                                                  \
    const TYPE r = cell[neighbor[i]];                                               \
    const TYPE phi = scheme == 0 ? (m >= (TYPE)0 ? l : r) : (TYPE)0.5 * (l + r);   \
    flux[i] = m * phi;                                                              \
}                                                                                  \
extern "C" __global__ void boundary_flux_##SUFFIX(                                  \
    const unsigned int* owner, const unsigned char* active,                         \
    const TYPE* cell, const TYPE* mass, const TYPE* boundary, TYPE* flux,            \
    unsigned int scheme, unsigned int offset, unsigned int count) {                 \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i >= count) return;                                                         \
    const unsigned int f = offset + i;                                              \
    if (!active[f]) { flux[f] = (TYPE)0; return; }                                  \
    const TYPE m = mass[f];                                                         \
    const TYPE l = cell[owner[i]];                                                  \
    const TYPE r = boundary[i];                                                     \
    const TYPE phi = scheme == 0 ? (m >= (TYPE)0 ? l : r) : (TYPE)0.5 * (l + r);   \
    flux[f] = m * phi;                                                              \
}                                                                                  \
extern "C" __global__ void cell_gather_##SUFFIX(                                    \
    const unsigned int* offsets, const unsigned int* faces, const signed char* signs,\
    const unsigned char* active, const TYPE* flux, TYPE* residual,                  \
    unsigned int count) {                                                          \
    const unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (cell >= count) return;                                                      \
    if (!active[cell]) { residual[cell] = (TYPE)0; return; }                        \
    TYPE sum = (TYPE)0;                                                             \
    for (unsigned int j = offsets[cell]; j < offsets[cell + 1]; ++j) {              \
        sum += (TYPE)signs[j] * flux[faces[j]];                                     \
    }                                                                              \
    residual[cell] = sum;                                                          \
}                                                                                  \
extern "C" __global__ void internal_value_##SUFFIX(                                \
    const unsigned int* owner, const unsigned int* neighbor, const TYPE* cell,       \
    TYPE* face_value, unsigned int count) {                                         \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < count) face_value[i] = (TYPE)0.5 * (cell[owner[i]] + cell[neighbor[i]]);\
}                                                                                  \
extern "C" __global__ void boundary_value_##SUFFIX(                                \
    const TYPE* boundary, TYPE* face_value, unsigned int offset,                    \
    unsigned int count) {                                                          \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < count) face_value[offset + i] = boundary[i];                            \
}                                                                                  \
extern "C" __global__ void gradient_gather_##SUFFIX(                               \
    const unsigned int* offsets, const unsigned int* faces, const signed char* signs,\
    const unsigned int* face_geometry, const double* nx, const double* ny,          \
    const double* nz, const double* volume, const unsigned char* face_active,       \
    const unsigned char* cell_active, const TYPE* face_value,                       \
    TYPE* gx, TYPE* gy, TYPE* gz, unsigned int count) {                             \
    const unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;                \
    if (cell >= count) return;                                                      \
    if (!cell_active[cell] || volume[cell] == 0.0) {                                \
        gx[cell] = gy[cell] = gz[cell] = (TYPE)0; return;                           \
    }                                                                              \
    double x = 0.0, y = 0.0, z = 0.0;                                              \
    for (unsigned int j = offsets[cell]; j < offsets[cell + 1]; ++j) {              \
        const unsigned int f = faces[j];                                            \
        if (!face_active[f]) continue;                                              \
        const unsigned int g = face_geometry[f];                                    \
        const double scale = (double)signs[j] * (double)face_value[f];              \
        x += scale * nx[g]; y += scale * ny[g]; z += scale * nz[g];                 \
    }                                                                              \
    gx[cell] = (TYPE)(x / volume[cell]);                                            \
    gy[cell] = (TYPE)(y / volume[cell]);                                            \
    gz[cell] = (TYPE)(z / volume[cell]);                                            \
}                                                                                  \
extern "C" __global__ void internal_diffusion_##SUFFIX(                            \
    const unsigned int* owner, const unsigned int* neighbor,                        \
    const unsigned int* geometry, const unsigned char* active,                      \
    const double* nx, const double* ny, const double* nz,                           \
    const double* cx, const double* cy, const double* cz,                           \
    const TYPE* cell, TYPE* flux, TYPE diffusivity, unsigned int count) {           \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i >= count) return;                                                         \
    if (!active[i]) { flux[i] = (TYPE)0; return; }                                  \
    const unsigned int l = owner[i], r = neighbor[i], g = geometry[i];              \
    const double dx = cx[r] - cx[l], dy = cy[r] - cy[l], dz = cz[r] - cz[l];        \
    const double d2 = dx*dx + dy*dy + dz*dz;                                        \
    const double nd = nx[g]*dx + ny[g]*dy + nz[g]*dz;                               \
    flux[i] = d2 == 0.0 ? (TYPE)0 : -diffusivity * (cell[r]-cell[l]) * (TYPE)(nd/d2);\
}                                                                                  \
extern "C" __global__ void boundary_diffusion_##SUFFIX(                            \
    const unsigned int* owner, const unsigned int* geometry,                        \
    const unsigned char* active, const double* nx, const double* ny, const double* nz,\
    const double* fx, const double* fy, const double* fz,                           \
    const double* cx, const double* cy, const double* cz,                           \
    const TYPE* cell, const TYPE* boundary, TYPE* flux, TYPE diffusivity,           \
    unsigned int offset, unsigned int count) {                                     \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i >= count) return;                                                         \
    const unsigned int f = offset+i;                                                \
    if (!active[f]) { flux[f] = (TYPE)0; return; }                                  \
    const unsigned int l = owner[i], g = geometry[i];                               \
    const double dx = fx[g]-cx[l], dy = fy[g]-cy[l], dz = fz[g]-cz[l];              \
    const double d2 = dx*dx + dy*dy + dz*dz;                                        \
    const double nd = nx[g]*dx + ny[g]*dy + nz[g]*dz;                               \
    flux[f] = d2 == 0.0 ? (TYPE)0 : -diffusivity * (boundary[i]-cell[l]) * (TYPE)(nd/d2);\
}                                                                                  \
extern "C" __global__ void fill_##SUFFIX(                                           \
    TYPE* output, TYPE value, unsigned int count) {                                 \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < count) output[i] = value;                                               \
}                                                                                  \
extern "C" __global__ void copy_##SUFFIX(                                           \
    const TYPE* input, TYPE* output, unsigned int count) {                          \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < count) output[i] = input[i];                                            \
}                                                                                  \
extern "C" __global__ void axpy_##SUFFIX(                                           \
    const TYPE* x, TYPE* y, TYPE alpha, unsigned int count) {                       \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < count) y[i] = alpha * x[i] + y[i];                                     \
}                                                                                  \
extern "C" __global__ void apply_mask_##SUFFIX(                                     \
    const unsigned char* mask, TYPE* values, unsigned int count) {                  \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                   \
    if (i < count && !mask[i]) values[i] = (TYPE)0;                                 \
}

DEFINE_FVM_KERNELS(float, f32)
DEFINE_FVM_KERNELS(double, f64)
