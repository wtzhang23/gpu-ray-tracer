#include "rayprimitives/entity.h"

namespace rprimitives {

CUDA_HOSTDEV
rmath::Vec3<float> Entity::point_to_local(const rmath::Vec3<float>& v) const {
    return o * (v - p);
}

CUDA_HOSTDEV
rmath::Vec3<float> Entity::point_from_local(const rmath::Vec3<float>& v) const {
    return (o.inverse() * v) + p;
}

CUDA_HOSTDEV
rmath::Vec3<float> Entity::vec_to_local(const rmath::Vec3<float>& v) const {
    return o * v;
}

CUDA_HOSTDEV
rmath::Vec3<float> Entity::vec_from_local(const rmath::Vec3<float>& v) const {
    return o.inverse() * v;
}

CUDA_HOSTDEV
rmath::Ray<float> Entity::ray_to_local(const rmath::Ray<float>& ray) const {
    rmath::Vec3<float> local_dir = vec_to_local(ray.direction());
    rmath::Vec3<float> local_pt = point_to_local(ray.origin());
    return rmath::Ray<float>(local_pt, local_dir);
}

CUDA_HOSTDEV
rmath::Ray<float> Entity::ray_from_local(const rmath::Ray<float>& ray) const {
    rmath::Vec3<float> nonlocal_dir = vec_from_local(ray.direction());
    rmath::Vec3<float> nonlocal_pt = point_from_local(ray.origin());
    return rmath::Ray<float>(nonlocal_pt, nonlocal_dir);
}

}