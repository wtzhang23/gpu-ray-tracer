#ifndef HITABLE_CUH
#define HITABLE_CUH

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayenv/scene.h"

namespace rprimitives {
struct Isect {
    bool hit;
    bool use_texture;
    float time;
    rmath::Vec3<float> norm;
    union Shade {
        rmath::Vec<float, 2> text_coords;
        rmath::Vec4<float> color;
        __device__
        Shade(){}
    } shading;
    Material mat;

    __device__
    Isect(): hit(false) {}
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
    virtual Isect hit_local(const rmath::Ray<float>& local_ray, renv::Scene& scene) {
        return Isect{};
    };
    
    __device__
    Isect hit(const rmath::Ray<float>& ray, renv::Scene& scene) {
        rmath::Ray<float> local_ray = rmath::Ray<float>(point_to_local(ray.origin()), vec_to_local(ray.direction()));
        Isect local_isect = hit_local(local_ray, scene);
        local_isect.norm = vec_from_local(local_isect.norm);
        return local_isect;
    }
};
}

#endif