#ifndef RAYPRIMITIVES_GPU_PHONG_CUH
#define RAYPRIMITIVES_GPU_PHONG_CUH

#include "raymath/geometry.h"

namespace renv {
namespace gpu {

class Scene;

}
}

namespace rprimitives {

class Isect;

namespace gpu {

__device__
rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& i, renv::gpu::Scene* s);

}
}

#endif