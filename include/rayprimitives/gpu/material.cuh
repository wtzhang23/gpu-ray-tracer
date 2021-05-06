#ifndef RAYPRIMITIVES_GPU_MATERIAL_CUH
#define RAYPRIMITIVES_GPU_MATERIAL_CUH

#include "rayprimitives/material.h"
#include "rayprimitives/gpu/hitable.cuh"

namespace renv {
namespace gpu {

class Scene;

}
}

namespace rprimitives {
namespace gpu {

__device__
rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& i, renv::gpu::Scene* s);

}
}

#endif