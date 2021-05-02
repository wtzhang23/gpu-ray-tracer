#ifndef SCENE_CUH
#define SCENE_CUH

#include "rayprimitives/hitable.cuh"
namespace renv {
__device__
bool cast_ray(Scene* scene, const rmath::Ray<float>& r, rprimitives::Isect& isect);

__device__
rmath::Vec4<float> propagate_ray(Scene* scene, const rmath::Ray<float>& r);

__device__
bool cast_local(Scene* scene, const rmath::Ray<float>& r, rprimitives::Isect& isect, const Transformation& t);
}

#endif