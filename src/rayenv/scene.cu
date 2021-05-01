#include "rayenv/scene.h"
#include "rayenv/scene.cuh"
namespace renv {
__device__
void cast_ray(Scene* scene, rmath::Ray<float> r, rprimitives::Isect& isect) {
    // TODO: use bvh tree
    rprimitives::Hitable** hitables = scene->get_hitables();
    for (int i = 0; i < scene->n_hitables(); i++) {
        rprimitives::Hitable* h = hitables[i];
        h->hit(r, scene, isect);
    }
}
}