#ifndef HITABLE_CUH
#define HITABLE_CUH

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayenv/scene.h"

namespace rprimitives {
struct Isect {
    float time;
    rmath::Vec3<float> norm;
    Shade* shading;
    rmath::Vec<float, 2> uv;
    Material* mat;

    __device__
    Isect(): time(INFINITY){}
};

class Hitable: public Entity {
public:
    __device__
    Hitable(): Entity() {}
    
    __device__
    Hitable(rmath::Vec3<float> position, rmath::Quat<float> orientation): Entity(position, orientation) {}
    
    __device__
    Hitable(Entity* entity): Entity(entity) {}

    CUDA_HOSTDEV
    virtual ~Hitable() {}
    
    __device__
    virtual bool hit_local(const rmath::Ray<float>& local_ray, renv::Scene* scene, Isect& isect) {
        isect = Isect{};
        return false;
    };
    
    __device__
    bool hit(const rmath::Ray<float>& ray, renv::Scene* scene, Isect& isect) {
        rmath::Vec3<float> local_dir = vec_to_local(ray.direction());
        float dir_len = local_dir.len();
        rmath::Ray<float> local_ray = rmath::Ray<float>(point_to_local(ray.origin()), local_dir);
        bool rv = hit_local(local_ray, scene, isect);
        if (rv) {
            isect.norm = vec_from_local(isect.norm);
            isect.time *= dir_len; // account for scaling
        }
        return rv;
    }
};
}

#endif