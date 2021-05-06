#ifndef RAYOPT_BOUNDING_BOX_H
#define RAYOPT_BOUNDING_BOX_H

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
class Entity;
}

namespace ropt {

struct BoundingBox {
    rmath::Vec3<float> min;
    rmath::Vec3<float> max;
    bool nondegenerate;

    CUDA_HOSTDEV
    BoundingBox(): min(), max(), nondegenerate(false) {}

    CUDA_HOSTDEV
    void fit_vertex(const rmath::Vec3<float>& v);

    CUDA_HOSTDEV
    friend BoundingBox& merge_into(BoundingBox& a, const BoundingBox& b);
    
    CUDA_HOSTDEV
    friend BoundingBox merge(const BoundingBox& a, const BoundingBox& b);
    
    CUDA_HOSTDEV
    friend BoundingBox from_local(const BoundingBox& a, const rprimitives::Entity& e);
    
    CUDA_HOSTDEV
    bool intersects(const rmath::Ray<float>& r, float& time) const;

    CUDA_HOSTDEV
    rmath::Vec3<float> center() const {
        return 0.5f * (min + max);
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> get_min() const {
        return min;
    }

    CUDA_HOSTDEV
    rmath::Vec3<float> get_max() const {
        return max;
    }

    CUDA_HOSTDEV
    bool is_degenerate() const {
        return !nondegenerate;
    }

    CUDA_HOSTDEV
    bool contains(const BoundingBox& other) const {
        if (!nondegenerate) {
            return false;
        } else if (!other.nondegenerate) {
            return true;
        }
        return min[0] <= other.min[0] && min[1] <= other.min[1] && min[2] && min[2] <= other.min[2]
                    && max[0] >= other.max[0] && max[1] >= other.max[1] && max[2] >= other.max[2];
    }

    CUDA_HOSTDEV
    float volume() const {
        rmath::Vec3<float> diff = max - min;
        return diff[0] * diff[1] * diff[2];
    }

    friend std::ostream& operator<<(std::ostream& os, const BoundingBox& box);
};

CUDA_HOSTDEV
BoundingBox& merge_into(BoundingBox& a, const BoundingBox& b);

CUDA_HOSTDEV
BoundingBox merge(const BoundingBox& a, const BoundingBox& b);

CUDA_HOSTDEV
BoundingBox from_local(const BoundingBox& a, const rprimitives::Entity& e);
}

#endif