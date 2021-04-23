#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace primitives {
}

#endif