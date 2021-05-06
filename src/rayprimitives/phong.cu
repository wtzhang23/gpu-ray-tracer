#include "rayenv/gpu/scene.h"
#include "rayenv/cpu/scene.h"
#include "rayprimitives/gpu/phong.cuh"
#include "rayprimitives/gpu/light.cuh"
#include "rayprimitives/cpu/phong.h"
#include "rayprimitives/cpu/light.h"
#include "rayprimitives/isect.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture_coords.h"
#include "raymath/linear.h"

namespace rprimitives {

__host__ __device__
rmath::Vec4<float> phong(const Isect& isect, rmath::Vec4<float> incoming_light, 
                const rmath::Vec3<float>& ray_dir, const rmath::Vec3<float>& dir_to_light) {
    // diffuse component
    rmath::Vec4<float> diffuse_col{};
    if (isect.coords->is_degenerate()) {
        diffuse_col = isect.mat->get_Kd();
    } else {
        // TODO: implement texture mapping
    }

    float norm_dot = max(rmath::dot(dir_to_light, isect.norm), 0.0f);
    rmath::Vec4<float> diffuse = norm_dot * isect.mat->get_Kd();
    
    // specular component
    rmath::Vec3<float> reflected = rmath::reflect(-dir_to_light, isect.norm);
    float reflect_dot = rmath::dot(-reflected, ray_dir);
    rmath::Vec4<float> specular = pow(max(reflect_dot, 0.0f), isect.mat->get_alpha()) * isect.mat->get_Ks();
    return (diffuse + specular) * incoming_light;
}

__host__ __device__
rmath::Vec4<float> org_light(const Isect& isect, const renv::Environment& env) {
    return isect.mat->get_Ke() + isect.mat->get_Ka() * env.get_ambience();
}

namespace gpu {

__device__
rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& isect, renv::gpu::Scene* s) {
    rprimitives::gpu::Light** lights = s->get_lights();
    rmath::Vec4<float> summed_colors = org_light(isect, s->get_environment());
    for (int i = 0; i < s->n_lights(); i++) {
        const rprimitives::gpu::Light* light = lights[i];
        rmath::Vec3<float> dir_to_light;
        rmath::Vec4<float> incoming_light = light->shine(org_ray.at(isect.time), dir_to_light, s);
        summed_colors += phong(isect, incoming_light, org_ray.direction(), dir_to_light);
    }
    return summed_colors;
}

}

namespace cpu {
    rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& isect, renv::cpu::Scene* s) {
        rmath::Vec4<float> summed_colors = org_light(isect, s->get_environment());
        for (const rprimitives::cpu::Light* light : s->get_lights()) {
            rmath::Vec3<float> dir_to_light;
            rmath::Vec4<float> incoming_light = light->shine(org_ray.at(isect.time), dir_to_light, s);
            summed_colors += phong(isect, incoming_light, org_ray.direction(), dir_to_light);
        }
        return summed_colors;
    }
}

}