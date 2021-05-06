#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayenv/canvas.h"
#include "rayenv/camera.h"
#include "rayenv/environment.h"
#include "rayopt/gpu/bvh.h"
#include "rayprimitives/gpu/texture.h"
#include "rayprimitives/gpu/vertex_buffer.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
namespace gpu {

class Hitable;
class Light;

}
}

namespace renv {
namespace gpu {

class Scene {
private:
    Environment env;
    rprimitives::gpu::Texture atlas;
    rprimitives::gpu::Hitable** hitables;
    rprimitives::gpu::Light** lights;
    rprimitives::gpu::VertexBuffer buffer;
    ropt::gpu::BVH bvh;
    int nh;
    int nl;
public:
    Scene(Environment env,
                rprimitives::gpu::Texture atlas, 
                rprimitives::gpu::Hitable** hitables, int n_hitables,
                rprimitives::gpu::Light** lights, int n_lights,
                rprimitives::gpu::VertexBuffer buffer): env(env),
                atlas(atlas), hitables(hitables), lights(lights),
                buffer(buffer), bvh(), nh(n_hitables), nl(n_lights) {}
    
    void set_bvh(ropt::gpu::BVH bvh) {
        this->bvh = bvh;
    }

    CUDA_HOSTDEV
    Environment& get_environment() {
        return env;
    }

    CUDA_HOSTDEV
    const ropt::gpu::BVH& get_bvh() const {
        return bvh;
    }

    CUDA_HOSTDEV
    rprimitives::gpu::Texture& get_atlas() {
        return atlas;
    }

    CUDA_HOSTDEV
    rprimitives::gpu::Hitable** get_hitables() {
        return hitables;
    }

    CUDA_HOSTDEV
    rprimitives::gpu::Light** get_lights() {
        return lights;
    }

    CUDA_HOSTDEV
    rprimitives::gpu::VertexBuffer& get_vertex_buffer() {
        return buffer;
    }

    CUDA_HOSTDEV
    int n_hitables() const {
        return nh;
    }

    CUDA_HOSTDEV
    int n_lights() const {
        return nl;
    }
};

}
}
#endif