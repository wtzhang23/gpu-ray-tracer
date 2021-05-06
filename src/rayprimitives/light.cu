#include "rayenv/gpu/scene.h"
#include "rayenv/gpu/scene.cuh"
#include "rayprimitives/gpu/light.cuh"
#include "rayprimitives/gpu/hitable.cuh"
#include "rayprimitives/material.h"

namespace rprimitives {
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

            // norm and shadow ray in same direction implies on inside of object
            if (rmath::dot(shadow_isect.norm, cur_shadow.direction()) > 0) {
                if (env.is_debugging()) {
                    printf("time: %f\nto_light_origin: (%f, %f, %f)\ncur_shadow_origin(%f, %f, %f)\ndir: (%f, %f, %f)\n", 
                            shadow_isect.time, to_light.origin()[0], to_light.origin()[1], to_light.origin()[2],
                            cur_shadow.origin()[0], cur_shadow.origin()[1], cur_shadow.origin()[2],
                            cur_shadow.direction()[0], cur_shadow.direction()[1], cur_shadow.direction()[2]);
                }
                const rmath::Vec4<float>& kt = shadow_isect.mat->get_Kt();
                if (kt[0] == 0 && kt[1] == 0 && kt[2] == 0 && kt[3] == 0) {
                    return rmath::Vec4<float>();
                }
                float atten_r = pow(kt[0], shadow_isect.time);
                float atten_g = pow(kt[1], shadow_isect.time);
                float atten_b = pow(kt[2], shadow_isect.time);
                float atten_a = pow(kt[3], shadow_isect.time);
                rv *= rmath::Vec4<float>({atten_r, atten_g, atten_b, atten_a});
            }
            cur_shadow = rmath::Ray<float>(cur_shadow.at(shadow_isect.time), cur_shadow.direction());
        } else {
            return rv;
        }
    }
}

__device__
rmath::Vec4<float> PointLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::gpu::Scene* scene) const {
    rmath::Vec3<float> light_disp = pos - hit_pos;
    float dist = light_disp.len();
    rmath::Vec3<float> dist_atten_consts = scene->get_environment().get_dist_atten();
    float quad = dist_atten_consts[0] + dist_atten_consts[1] * dist + dist_atten_consts[2] * dist * dist;
    float dist_atten = quad < 1.0f ? 1.0f : 1.0f / quad; 
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
}