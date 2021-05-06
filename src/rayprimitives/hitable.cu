#include "rayprimitives/gpu/hitable.cuh"
#include "rayprimitives/cpu/hitable.h"
#include "rayprimitives/entity.h"

namespace rprimitives {

struct HitHandle {
    const rmath::Ray<float>& ray;
    Isect& isect;
    float scale;

    CUDA_HOSTDEV
    HitHandle(const rmath::Ray<float>& ray, Isect& isect): ray(ray), isect(isect), scale(1) {}
    
    CUDA_HOSTDEV
    rmath::Ray<float> get_local_ray(const Entity& entity) {
        scale = entity.vec_to_local(ray.direction()).len();
        return entity.ray_to_local(ray);
    }

    CUDA_HOSTDEV
    void fix_isect(const Entity& entity) const {
        isect.norm = entity.vec_from_local(isect.norm).normalized();
        isect.time *= scale;
    }
};

namespace gpu {
__device__
bool Hitable::hit(const rmath::Ray<float>& ray, renv::gpu::Scene* scene, Isect& isect) {
    HitHandle handle{ray, isect};
    rmath::Ray<float> local_ray = handle.get_local_ray(*this);
    bool rv = hit_local(local_ray, scene, isect);
    if (rv) {
        handle.fix_isect(*this);
    }
    return rv;
}
}

namespace cpu {
bool Hitable::hit(const rmath::Ray<float>& ray, renv::cpu::Scene* scene, Isect& isect) const {
    HitHandle handle{ray, isect};
    rmath::Ray<float> local_ray = handle.get_local_ray(*this);
    bool rv = hit_local(local_ray, scene, isect);
    if (rv) {
        handle.fix_isect(*this);
    }
    return rv;
}
}

}