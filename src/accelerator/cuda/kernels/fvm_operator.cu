// Multi-component, component-major FVM operator kernels.
__device__ double mesh_limiter(unsigned int limiter, double up, double dn) {
    if (limiter == 0) return 1.0;
    const double diff = dn - up;
    const double r = fabs(diff) < 1.0e-14 ? 1.0 : fabs((up - dn) / diff);
    if (limiter == 1) return fmin(fmax(r, 0.0), 1.0);
    if (limiter == 2) return (r + fabs(r)) / (1.0 + fabs(r));
    return fmax(fmax(fmin(2.0*r, 1.0), fmin(r, 2.0)), 0.0);
}

__device__ double mesh_face_value(double l, double r, double mass,
                                  unsigned int scheme, double blend,
                                  unsigned int scheme_limiter,
                                  unsigned int reconstruction_limiter) {
    const double up = mass >= 0.0 ? l : r;
    const double dn = mass >= 0.0 ? r : l;
    const double central = 0.5 * (l + r);
    const double b = fmin(fmax(blend, 0.0), 1.0);
    const double lo = fmin(l, r), hi = fmax(l, r);
    if (scheme == 0) return up;
    if (scheme == 1) return central;
    if (scheme == 2) return fmin(fmax(up + b * (central-up), lo), hi);
    if (scheme == 3) return (1.0-b)*up + b*central;
    const double psi = mesh_limiter(scheme_limiter, up, dn)
                     * mesh_limiter(reconstruction_limiter, up, dn);
    const double high = up + 0.5 * psi * (dn-up);
    return fmin(fmax((1.0-b)*up + b*high, lo), hi);
}

__device__ double mesh_exterior(unsigned char kind, double alpha, double beta,
                                double gamma, double inside, double dirichlet) {
    if (kind == 0) return dirichlet;
    if (kind == 1 || fabs(alpha) < 1.0e-14) return inside;
    return (gamma - beta * inside) / alpha;
}

#define DEFINE_OPERATOR(TYPE, SUFFIX)                                               \
extern "C" __global__ void op_internal_value_##SUFFIX(                             \
 const unsigned int* owner, const unsigned int* neighbor, const TYPE* cell,         \
 TYPE* face_value, unsigned int cells, unsigned int faces, unsigned int count,      \
 unsigned int components) {                                                         \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x, total=count*components;   \
  if(k>=total)return; const unsigned int c=k/count,i=k%count;                       \
  face_value[c*faces+i]=(TYPE)(0.5*((double)cell[c*cells+owner[i]]                  \
                                  +(double)cell[c*cells+neighbor[i]]));             \
}                                                                                   \
extern "C" __global__ void op_boundary_value_##SUFFIX(                             \
 const unsigned int* owner, const unsigned char* kind, const double* alpha,         \
 const double* beta, const double* gamma, const TYPE* cell,const TYPE* boundary,    \
 TYPE* face_value,                                                                  \
 unsigned int cells, unsigned int faces, unsigned int offset, unsigned int count,   \
 unsigned int components) {                                                         \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x,total=count*components;    \
  if(k>=total)return; const unsigned int c=k/count,i=k%count;                       \
  const double inside=(double)cell[c*cells+owner[i]];                               \
  face_value[c*faces+offset+i]=(TYPE)mesh_exterior(kind[i],alpha[i],beta[i],gamma[i],inside,(double)boundary[c*count+i]);\
}                                                                                   \
extern "C" __global__ void op_green_gauss_##SUFFIX(                                \
 const unsigned int* offsets,const unsigned int* face_index,const signed char* signs,\
 const unsigned int* face_geometry,const double* nx,const double* ny,const double* nz,\
 const double* volume,const unsigned char* face_active,const unsigned char* cell_active,\
 const TYPE* face_value,TYPE* gx,TYPE* gy,TYPE* gz,unsigned int cells,               \
 unsigned int faces,unsigned int components) {                                      \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x,total=cells*components;    \
  if(k>=total)return; const unsigned int c=k/cells,cell=k%cells;                    \
  if(!cell_active[cell]){gx[k]=gy[k]=gz[k]=(TYPE)0;return;}                         \
  double x=0.0,y=0.0,z=0.0;                                                        \
  for(unsigned int j=offsets[cell];j<offsets[cell+1];++j){                          \
   const unsigned int f=face_index[j];if(!face_active[f])continue;                 \
   const unsigned int g=face_geometry[f];                                           \
   const double scale=(double)signs[j]*(double)face_value[c*faces+f];               \
   x+=scale*nx[g];y+=scale*ny[g];z+=scale*nz[g];                                    \
  } gx[k]=(TYPE)(x/volume[cell]);gy[k]=(TYPE)(y/volume[cell]);gz[k]=(TYPE)(z/volume[cell]);\
}                                                                                   \
extern "C" __global__ void op_least_squares_##SUFFIX(                              \
 const unsigned int* offsets,const unsigned int* face_index,const unsigned int* neighbor,\
 const double* wx,const double* wy,const double* wz,const unsigned char* fallback,  \
 const unsigned char* face_active,const unsigned char* cell_active,                 \
 const unsigned int* boundary_owner,const unsigned char* kind,const double* alpha, \
 const double* beta,const double* gamma,const TYPE* cell,const TYPE* boundary,       \
 TYPE* gx,TYPE* gy,TYPE* gz,unsigned int cells,unsigned int internal_faces,          \
 unsigned int boundary_count,unsigned int components,                               \
 unsigned int preserve_fallback) {                                                  \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x,total=cells*components;    \
  if(k>=total)return;const unsigned int c=k/cells,ci=k%cells;                       \
  if(preserve_fallback&&fallback[ci])return;                                         \
  if(!cell_active[ci]){gx[k]=gy[k]=gz[k]=(TYPE)0;return;}                           \
  const double center=(double)cell[k];double x=0.0,y=0.0,z=0.0;                    \
  for(unsigned int j=offsets[ci];j<offsets[ci+1];++j){                             \
   const unsigned int f=face_index[j];if(!face_active[f])continue;                 \
   const unsigned int n=neighbor[j];double sample;                                  \
   if(n==0xffffffffu){const unsigned int b=f-internal_faces;                        \
    sample=mesh_exterior(kind[b],alpha[b],beta[b],gamma[b],center,(double)boundary[c*boundary_count+b]);\
   }else sample=(double)cell[c*cells+n];                                            \
   const double diff=sample-center;x+=wx[j]*diff;y+=wy[j]*diff;z+=wz[j]*diff;       \
  }gx[k]=(TYPE)x;gy[k]=(TYPE)y;gz[k]=(TYPE)z;                                       \
}                                                                                   \
extern "C" __global__ void op_internal_flux_##SUFFIX(                              \
 const unsigned int* owner,const unsigned int* neighbor,const unsigned int* geometry,\
 const unsigned char* active,const double* nx,const double* ny,const double* nz,    \
 const double* fx,const double* fy,const double* fz,const double* cx,const double* cy,\
 const double* cz,const TYPE* cell,const TYPE* mass,const TYPE* gx,const TYPE* gy,   \
 const TYPE* gz,const TYPE* face_source,TYPE* flux,TYPE* deferred,                  \
 unsigned int scheme,double blend,unsigned int scheme_limiter,                     \
 unsigned int reconstruction_limiter,unsigned int reconstruct,double diffusivity, \
 unsigned int diffusion_mode,unsigned int cells,unsigned int faces,                \
 unsigned int count,unsigned int components) {                                      \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x,total=count*components;    \
  if(k>=total)return;const unsigned int c=k/count,i=k%count,o=owner[i],n=neighbor[i],g=geometry[i];\
  const unsigned int out=c*faces+i;if(!active[i]){flux[out]=deferred[out]=(TYPE)0;return;}\
  double l=(double)cell[c*cells+o],r=(double)cell[c*cells+n];                       \
  if(reconstruct){l+=(double)gx[c*cells+o]*(fx[g]-cx[o])+(double)gy[c*cells+o]*(fy[g]-cy[o])+(double)gz[c*cells+o]*(fz[g]-cz[o]);\
   r+=(double)gx[c*cells+n]*(fx[g]-cx[n])+(double)gy[c*cells+n]*(fy[g]-cy[n])+(double)gz[c*cells+n]*(fz[g]-cz[n]);}\
  const double m=(double)mass[i];const double conv=m*mesh_face_value(l,r,m,scheme,blend,scheme_limiter,reconstruction_limiter);\
  const double dx=cx[n]-cx[o],dy=cy[n]-cy[o],dz=cz[n]-cz[o],d2=dx*dx+dy*dy+dz*dz;   \
  const double nd=nx[g]*dx+ny[g]*dy+nz[g]*dz;                                      \
  const double orth=d2>0.0?-diffusivity*((double)cell[c*cells+n]-(double)cell[c*cells+o])*nd/d2:0.0;\
  const double scale=d2>0.0?nd/d2:0.0;                                              \
  const double agx=0.5*((double)gx[c*cells+o]+(double)gx[c*cells+n]);               \
  const double agy=0.5*((double)gy[c*cells+o]+(double)gy[c*cells+n]);               \
  const double agz=0.5*((double)gz[c*cells+o]+(double)gz[c*cells+n]);               \
  const double nonorth=-diffusivity*(agx*(nx[g]-dx*scale)+agy*(ny[g]-dy*scale)+agz*(nz[g]-dz*scale));\
  const double diff=diffusion_mode==2?orth+nonorth:orth;                            \
  deferred[out]=(TYPE)(diffusion_mode==1?-nonorth:0.0);                             \
  flux[out]=(TYPE)(conv+diff+(double)face_source[out]);                             \
}                                                                                   \
extern "C" __global__ void op_boundary_flux_##SUFFIX(                              \
 const unsigned int* owner,const unsigned int* geometry,const unsigned char* active,\
 const double* area,const double* fx,const double* fy,const double* fz,             \
 const double* cx,const double* cy,const double* cz,const unsigned char* conv_kind, \
 const double* conv_alpha,const double* conv_beta,const double* conv_gamma,         \
 const unsigned char* diff_kind,const double* diff_alpha,const double* diff_beta,   \
 const double* diff_gamma,const TYPE* cell,const TYPE* boundary,const TYPE* mass,   \
 const TYPE* gx,const TYPE* gy,const TYPE* gz,const TYPE* face_source,TYPE* flux,    \
 TYPE* deferred,                                                                   \
 unsigned int scheme,double blend,unsigned int scheme_limiter,                     \
 unsigned int reconstruction_limiter,unsigned int reconstruct,double diffusivity, \
 unsigned int cells,unsigned int faces,unsigned int offset,unsigned int count,     \
 unsigned int components) {                                                         \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x,total=count*components;    \
  if(k>=total)return;const unsigned int c=k/count,i=k%count,o=owner[i],g=geometry[i],f=offset+i;\
  const unsigned int out=c*faces+f;if(!active[f]){flux[out]=deferred[out]=(TYPE)0;return;}\
  double inside=(double)cell[c*cells+o];                                            \
  if(reconstruct)inside+=(double)gx[c*cells+o]*(fx[g]-cx[o])+(double)gy[c*cells+o]*(fy[g]-cy[o])+(double)gz[c*cells+o]*(fz[g]-cz[o]);\
  const double exterior=mesh_exterior(conv_kind[i],conv_alpha[i],conv_beta[i],conv_gamma[i],inside,(double)boundary[c*count+i]);\
  const double m=(double)mass[f];const double conv=m*mesh_face_value(inside,exterior,m,scheme,blend,scheme_limiter,reconstruction_limiter);\
  double diff,source=0.0;if(diff_kind[i]==0){const double dx=fx[g]-cx[o],dy=fy[g]-cy[o],dz=fz[g]-cz[o];\
    const double distance=fmax(sqrt(dx*dx+dy*dy+dz*dz),1.0e-14);                    \
    diff=-diffusivity*((double)boundary[c*count+i]-(double)cell[c*cells+o])/distance*area[g];\
  }else if(diff_kind[i]==1)diff=-diffusivity*diff_gamma[i]*area[g];                \
  else {const double b=fmax(diff_beta[i],1.0e-14);                                 \
    diff=-diffusivity*(diff_gamma[i]-diff_alpha[i]*(double)cell[c*cells+o])/b*area[g];\
    source=diffusivity*diff_gamma[i]*area[g]/b;}                                   \
  deferred[out]=(TYPE)source;flux[out]=(TYPE)(conv+diff+(double)face_source[out]);  \
}                                                                                   \
extern "C" __global__ void op_cell_gather_##SUFFIX(                                \
 const unsigned int* offsets,const unsigned int* face_index,const signed char* signs,\
 const unsigned char* active,const TYPE* flux,const TYPE* deferred,const TYPE* cell_source,\
 TYPE* residual,unsigned int cells,unsigned int faces,unsigned int components) {    \
  const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x,total=cells*components;    \
  if(k>=total)return;const unsigned int c=k/cells,cell=k%cells;if(!active[cell]){residual[k]=(TYPE)0;return;}\
  double sum=(double)cell_source[k];for(unsigned int j=offsets[cell];j<offsets[cell+1];++j){\
   const unsigned int f=face_index[j];const double sign=(double)signs[j];            \
   sum+=sign*(double)flux[c*faces+f];if(sign>0.0)sum+=(double)deferred[c*faces+f];   \
  }residual[k]=(TYPE)sum;                                                           \
}

DEFINE_OPERATOR(float, f32)
DEFINE_OPERATOR(double, f64)
