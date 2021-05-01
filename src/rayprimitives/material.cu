#include "rayprimitives/material.cuh"
#include "rayenv/scene.h"
#include "rayprimitives/light.cuh"

namespace rprimitives {

__device__
rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& isect, renv::Scene* s) {
    rprimitives::Light** lights = s->get_lights();
    rmath::Vec4<float> summed_colors = isect.mat->get_Ke() + isect.mat->get_Ka() * s->get_ambience();
    for (int i = 0; i < s->n_lights(); i++) {
        rprimitives::Light* light = lights[i];
        rmath::Vec3<float> dir_to_light;
        rmath::Vec4<float> incoming_light = light->shine(org_ray.at(isect.time), dir_to_light, s);
        
        // diffuse component
        float norm_dot = abs(rmath::dot(dir_to_light, isect.norm));
        rmath::Vec4<float> diffuse = norm_dot * isect.mat->get_Kd();
        
        // specular component
        rmath::Vec3<float> reflected = -rmath::reflect(-dir_to_light, isect.norm);
        float reflect_dot = rmath::dot(reflected, org_ray.direction());
        rmath::Vec4<float> specular = pow(max(reflect_dot, 0.0f), isect.mat->get_alpha()) * isect.mat->get_Ks();
        summed_colors += (diffuse + specular) * incoming_light;
    }
    return summed_colors;
}

}