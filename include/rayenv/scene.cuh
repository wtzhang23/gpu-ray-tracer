#ifndef SCENE_CUH
#define SCENE_CUH

#include "rayprimitives/hitable.cuh"
namespace renv {
__device__
void cast_ray(Scene* scene, rmath::Ray<float> r, rprimitives::Isect& isect);
}

#endif