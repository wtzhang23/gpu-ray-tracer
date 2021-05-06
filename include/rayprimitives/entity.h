#ifndef RAYPRIMITIVES_ENTITY_H
#define RAYPRIMITIVES_ENTITY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "raymath/linear.h"
#include "raymath/geometry.h"

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
    CUDA_HOSTDEV
    Entity(rmath::Vec3<float> position, rmath::Quat<float> orientation): o(orientation), p(position) {}
    
    CUDA_HOSTDEV
    Entity(): o(rmath::Quat<float>::identity()), p(){}
    
    CUDA_HOSTDEV
    Entity(const Entity* entity): o(entity->o), p(entity->p) {}

    CUDA_HOSTDEV
    rmath::Vec3<float> point_to_local(const rmath::Vec3<float>& v) const;

    CUDA_HOSTDEV
    rmath::Vec3<float> point_from_local(const rmath::Vec3<float>& v) const;

    CUDA_HOSTDEV
    rmath::Vec3<float> vec_to_local(const rmath::Vec3<float>& v) const;

    CUDA_HOSTDEV
    rmath::Vec3<float> vec_from_local(const rmath::Vec3<float>& v) const;

    CUDA_HOSTDEV
    rmath::Ray<float> ray_to_local(const rmath::Ray<float>& ray) const;

    CUDA_HOSTDEV
    rmath::Ray<float> ray_from_local(const rmath::Ray<float>& ray) const;

    CUDA_HOSTDEV
    void translate_global(rmath::Vec3<float> dp) {
        p += dp;
    }

    CUDA_HOSTDEV
    void translate(rmath::Vec3<float> dp) {
        translate_global(vec_to_local(dp));
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> pos() const {
        return this->p;
    }

    CUDA_HOSTDEV
    void rotate(rmath::Quat<float> dr) {
        o = dr * o;
    }

    CUDA_HOSTDEV
    void set_position(rmath::Vec3<float> p) {
        this->p = p;
    }

    CUDA_HOSTDEV
    void set_orientation(rmath::Quat<float> o) {
        this->o = o;
    }
};
}

#endif