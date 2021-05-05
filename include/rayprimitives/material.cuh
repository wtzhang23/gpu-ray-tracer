#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "rayprimitives/material.h"
#include "rayprimitives/hitable.cuh"

namespace renv {
class Scene;
}

namespace rprimitives {

__device__
rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& i, renv::Scene* s);

}

#endif