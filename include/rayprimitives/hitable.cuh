#ifndef HITABLE_CUH
#define HITABLE_CUH

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayenv/scene.h"

namespace rprimitives {
struct Shade {
    union Data {
        struct TextData {
            int texture_x;
            int texture_y;
            int texture_width;
            int texture_height;
        } text_data;
        rmath::Vec4<float> col;
        __host__ __device__
        Data(rmath::Vec4<float> col): col(col) {}
        __host__ __device__
        Data(int texture_x, int texture_y, int texture_width, int texture_height): text_data{texture_x, texture_y, texture_width, texture_height}{}
    } data;
    bool use_texture;
    __host__ __device__
    Shade(rmath::Vec4<float> col): data(col), use_texture(false) {}
    __host__ __device__
    Shade(int texture_x, int texture_y, int texture_width, int texture_height): data(texture_x, texture_y, texture_width, texture_height),
                            use_texture(true) {}
};

struct Isect {
    bool hit;
    float time;
    rmath::Vec3<float> norm;
    Shade* shading;
    rmath::Vec<float, 2> uv;
    Material* mat;

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
    virtual bool hit_local(const rmath::Ray<float>& local_ray, renv::Scene* scene, Isect& isect) {
        isect = Isect{};
        return false;
    };
    
    __device__
    bool hit(const rmath::Ray<float>& ray, renv::Scene* scene, Isect& isect) {
        rmath::Vec3<float> local_dir = vec_to_local(ray.direction());
        float dir_len = local_dir.len();
        rmath::Ray<float> local_ray = rmath::Ray<float>(point_to_local(ray.origin()), local_dir);
        if (hit_local(local_ray, scene, isect)) {
            isect.norm = vec_from_local(isect.norm);
            isect.time *= dir_len; // account for scaling
            return true;
        }
        return false;
    }
};
}

#endif