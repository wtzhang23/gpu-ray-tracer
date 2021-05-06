#include "rayenv/gpu/scene.h"
#include "rayenv/gpu/scene.cuh"
#include "rayenv/cpu/scene.h"
#include "rayprimitives/gpu/light.cuh"
#include "rayprimitives/cpu/light.h"
#include "rayprimitives/gpu/hitable.cuh"
#include "rayprimitives/material.h"

namespace rprimitives {

CUDA_HOSTDEV
float calc_dist_atten(const renv::Environment& env, float dist) {
    rmath::Vec3<float> dist_atten_consts = env.get_dist_atten();
    float quad = dist_atten_consts[0] + dist_atten_consts[1] * dist + dist_atten_consts[2] * dist * dist;
    float dist_atten = quad < 1.0f ? 1.0f : 1.0f / quad; 
    return dist_atten;
}

CUDA_HOSTDEV
rmath::Vec4<float> calc_shadow_atten(const rmath::Vec4<float>& kt, float dist) {
    float atten_r = pow(kt[0], dist);
    float atten_g = pow(kt[1], dist);
    float atten_b = pow(kt[2], dist);
    float atten_a = pow(kt[3], dist);
    return rmath::Vec4<float>({atten_r, atten_g, atten_b, atten_a});
}

namespace gpu {
__device__
rmath::Vec4<float> Light::attenuate(const rmath::Ray<float>& to_light, float max_t, renv::gpu::Scene* scene) const {
    rmath::Vec4<float> rv = color;
    rmath::Ray<float> cur_shadow = rmath::Ray<float>(to_light.at(rmath::THRESHOLD), to_light.direction());
    renv::Environment& env = scene->get_environment();
    while (true) {
        float time = INFINITY;
        rprimitives::Isect shadow_isect{time};

        if (env.is_debugging()) {
            printf("shooting shadow ray\n");
        }

        if (renv::gpu::cast_ray(scene, cur_shadow, shadow_isect)) {
            if (shadow_isect.time > max_t) {
                return rv;
            }
            if (!shadow_isect.mat->refractive()) {
                return rmath::Vec4<float>();
            }

            // norm and shadow ray in same direction implies on inside of object
            if (rmath::dot(shadow_isect.norm, cur_shadow.direction()) > 0) {
                const rmath::Vec4<float>& kt = shadow_isect.mat->get_Kt();
                rv *= calc_shadow_atten(kt, shadow_isect.time);
            }
            cur_shadow = rmath::Ray<float>(cur_shadow.at(shadow_isect.time), cur_shadow.direction());
            max_t -= shadow_isect.time;
        } else {
            return rv;
        }
    }
}

__device__
rmath::Vec4<float> PointLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::gpu::Scene* scene) const {
    rmath::Vec3<float> light_disp = pos - hit_pos;
    float dist = light_disp.len();
    float dist_atten = calc_dist_atten(scene->get_environment(), dist);
    dir_to_light = light_disp.normalized();
    rmath::Ray<float> to_light = rmath::Ray<float>(hit_pos, dir_to_light);
    return dist_atten * attenuate(to_light, dist, scene);
}

__device__
rmath::Vec4<float> DirLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::gpu::Scene* scene) const {
    dir_to_light = -dir;
    return attenuate(rmath::Ray<float>(hit_pos, dir_to_light), INFINITY, scene);
}
}

namespace cpu {
rmath::Vec4<float> Light::attenuate(const rmath::Ray<float>& to_light, float max_t, renv::cpu::Scene* scene) const {
    rmath::Vec4<float> rv = color;
    rmath::Ray<float> cur_shadow = rmath::Ray<float>(to_light.at(rmath::THRESHOLD), to_light.direction());
    renv::Environment& env = scene->get_environment();

    while (true) {
        float time = INFINITY;
        rprimitives::Isect shadow_isect{time};

        if (env.is_debugging()) {
            printf("shooting shadow ray\n");
        }

        if (scene->cast_ray(cur_shadow, shadow_isect)) {
            if (shadow_isect.time > max_t) {
                return rv;
            }

            if (!shadow_isect.mat->refractive()) {
                return rmath::Vec4<float>();
            }

            // norm and shadow ray in same direction implies on inside of object
            if (rmath::dot(shadow_isect.norm, cur_shadow.direction()) > 0) {
                const rmath::Vec4<float>& kt = shadow_isect.mat->get_Kt();
                rv *= calc_shadow_atten(kt, shadow_isect.time);
            }
            cur_shadow = rmath::Ray<float>(cur_shadow.at(shadow_isect.time), cur_shadow.direction());
            max_t -= shadow_isect.time;
        } else {
            return rv;
        }
    }
}

rmath::Vec4<float> PointLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::cpu::Scene* scene) const {
    rmath::Vec3<float> light_disp = pos - hit_pos;
    float dist = light_disp.len();
    float dist_atten = calc_dist_atten(scene->get_environment(), dist);
    dir_to_light = light_disp.normalized();
    rmath::Ray<float> to_light = rmath::Ray<float>(hit_pos, dir_to_light);
    return dist_atten * attenuate(to_light, dist, scene);
}

rmath::Vec4<float> DirLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::cpu::Scene* scene) const {
    dir_to_light = -dir;
    return attenuate(rmath::Ray<float>(hit_pos, dir_to_light), INFINITY, scene);
}
}
}