#ifndef MATERIAL_CUH
#define MATERIAL_CUH

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