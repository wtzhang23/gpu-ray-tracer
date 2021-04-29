#ifndef ENTITY_H
#define ENTITY_H

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
class Entity {
protected:
    rmath::Quat<float> o;
    rmath::Vec3<float> p;
public:
    Entity(rmath::Vec3<float> position, rmath::Quat<float> orientation): o(orientation), p(position) {}
    Entity(): o(rmath::Quat<float>::identity()), p(){}

    CUDA_HOSTDEV
    rmath::Vec3<float> point_to_local(const rmath::Vec3<float>& v) const {
        return o * (v - p);
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> point_from_local(const rmath::Vec3<float>& v) const {
        return (o.inverse() * v) + p;
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> vec_to_local(const rmath::Vec3<float>& v) const {
        return o * v;
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> vec_from_local(const rmath::Vec3<float>& v) const {
        return o.inverse() * v;
    }

    void translate_global(rmath::Vec3<float> dp) {
        p += dp;
    }

    void translate(rmath::Vec3<float> dp) {
        translate_global(vec_to_local(dp));
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> pos() const {
        return this->p;
    }

    void rotate(rmath::Quat<float> dr) {
        o = dr * o;
    }

    void set_position(rmath::Vec3<float> p) {
        this->p = p;
    }

    void set_orientation(rmath::Quat<float> o) {
        this->o = o;
    }
};

struct Isect {
    bool hit;
    bool use_texture;
    float time;
    union Shade {
        rmath::Vec<float, 2> text_coords;
        rmath::Vec4<float> color;
        CUDA_HOSTDEV
        Shade(){}
    } shading;
    Material mat;

    CUDA_HOSTDEV
    Isect() {}
};

class Hitable: public Entity {
public:
    Hitable(): Entity() {}
    Hitable(rmath::Vec3<float> position, rmath::Quat<float> orientation): Entity(position, orientation) {}
    CUDA_HOSTDEV
    Hitable(Entity entity): Entity(entity) {}

    __device__
    virtual Isect hit_local(const rmath::Ray<float>& local_ray) {
        return Isect{};
    };
    
    __device__
    Isect hit(const rmath::Ray<float>& ray) {
        rmath::Ray<float> local_ray = rmath::Ray<float>(point_to_local(ray.origin()), vec_to_local(ray.direction()));
        return hit_local(local_ray);
    }
};
}

#endif