#ifndef RAYENV_CPU_SCENE_H
#define RAYENV_CPU_SCENE_H

#include <vector>
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayenv/canvas.h"
#include "rayenv/camera.h"
#include "rayenv/environment.h"
#include "rayprimitives/cpu/texture.h"
#include "rayprimitives/cpu/vertex_buffer.h"
#include "rayopt/cpu/bvh.h"

namespace rprimitives {

class Isect;

namespace cpu {
    class Hitable;
    class Light;
}

}

namespace renv {
namespace cpu {

class Scene {
private:
    Environment env;
    rprimitives::cpu::Texture atlas;
    std::vector<rprimitives::cpu::Hitable*> hitables;
    std::vector<rprimitives::cpu::Light*> lights;
    rprimitives::cpu::VertexBuffer buffer;
    ropt::cpu::BVH bvh;

    rmath::Vec4<float> propagate_helper(const rmath::Ray<float>& r, rprimitives::Isect& isect, float& last_time, bool in_obj, int depth);

public:
    Scene(Environment env,
            rprimitives::cpu::Texture atlas,
            std::vector<rprimitives::cpu::Hitable*> hitables,
            std::vector<rprimitives::cpu::Light*> lights,
            rprimitives::cpu::VertexBuffer buffer): env(env),
            atlas(atlas), hitables(hitables), lights(lights),
            buffer(buffer), bvh(){}
    ~Scene();
    Scene(const Scene& scene) = delete;
    Scene(Scene&& scene);
    Scene& operator=(const Scene& scene) = delete;
    Scene& operator=(Scene&& scene);

    const ropt::cpu::BVH& get_bvh() const {
        return bvh;
    }

    void set_bvh(ropt::cpu::BVH bvh) {
        this->bvh = bvh;
    }

    Environment& get_environment() {
        return env;
    }

    rprimitives::cpu::Texture& get_atlas() {
        return atlas;
    }

    std::vector<rprimitives::cpu::Hitable*>& get_hitables() {
        return hitables;
    }

    std::vector<rprimitives::cpu::Light*>& get_lights() {
        return lights;
    }

    rprimitives::cpu::VertexBuffer& get_vertex_buffer() {
        return buffer;
    }

    bool cast_ray(const rmath::Ray<float>& r, rprimitives::Isect& isect);
    rmath::Vec4<float> propagate_ray(const rmath::Ray<float>& r, rprimitives::Isect& isect);
    bool cast_local(const rmath::Ray<float>& r, rprimitives::Isect& isect, const Transformation& t);
};

}
}

#endif