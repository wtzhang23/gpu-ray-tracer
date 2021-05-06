#ifndef RAYOPT_ZORDER_H
#define RAYOPT_ZORDER_H

#include "raymath/linear.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace ropt {
CUDA_HOSTDEV
unsigned long z_order(rmath::Vec3<float> vec);
}

#endif