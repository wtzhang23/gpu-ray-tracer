#include "rayprimitives/light.cuh"
#include "rayenv/scene.h"
#include "rayenv/scene.cuh"

namespace rprimitives {
__device__
rmath::Vec4<float> Light::attenuate(const rmath::Ray<float>& to_light, float max_t, renv::Scene* scene) const {
    bool in_obj = false;
    rmath::Vec4<float> rv = color;
    rmath::Ray<float> cur_shadow = rmath::Ray<float>(to_light.at(rmath::THRESHOLD), to_light.direction());
    while (true) {
        rprimitives::Isect shadow_isect{};
        if (renv::cast_ray(scene, cur_shadow, shadow_isect)) {
            if (shadow_isect.time > max_t) {
                return rv;
            }

            if (in_obj) {
                const rmath::Vec4<float>& kt = shadow_isect.mat->get_Kt();
                float atten_r = pow(kt[0], shadow_isect.time);
                float atten_g = pow(kt[1], shadow_isect.time);
                float atten_b = pow(kt[2], shadow_isect.time);
                float atten_a = pow(kt[3], shadow_isect.time);
                rv *= rmath::Vec4<float>({atten_r, atten_g, atten_b, atten_a});
            }
            in_obj = !in_obj;
            cur_shadow = rmath::Ray<float>(cur_shadow.at(shadow_isect.time + rmath::THRESHOLD), cur_shadow.direction());
        } else {
            return rv;
        }
    }
}

__device__
rmath::Vec4<float> PointLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::Scene* scene) const {
    rmath::Vec3<float> light_disp = pos - hit_pos;
    float dist = light_disp.len();
    rmath::Vec3<float> dist_atten_consts = scene->get_dist_atten();
    float quad = dist_atten_consts[0] + dist_atten_consts[1] * dist + dist_atten_consts[2] * dist * dist;
    float dist_atten = quad < 1.0f ? 1.0f : 1.0f / quad; 
    dir_to_light = light_disp.normalized();
    rmath::Ray<float> to_light = rmath::Ray<float>(hit_pos, dir_to_light);
    return dist_atten * attenuate(to_light, dist, scene);
}

__device__
rmath::Vec4<float> DirLight::shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::Scene* scene) const {
    dir_to_light = -dir;
    return attenuate(rmath::Ray<float>(hit_pos, dir_to_light), INFINITY, scene);
}
}