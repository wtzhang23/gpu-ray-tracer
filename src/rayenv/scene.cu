#include "rayenv/scene.h"
#include "rayenv/scene.cuh"
#include "rayprimitives/material.h"
#include "rayprimitives/material.cuh"
namespace renv {
static const int MAX_DEPTH = 10;

__device__
bool cast_local(Scene* scene, const rmath::Ray<float>& r, rprimitives::Isect& isect, const Transformation& t) {
    rprimitives::Hitable** hitables = scene->get_hitables();
    rprimitives::Hitable* h = hitables[t.get_hitable_idx()];
    
    rmath::Vec3<float> local_dir = t.vec_to_local(r.direction());
    float dir_len = local_dir.len();
    rmath::Ray<float> local_ray = rmath::Ray<float>({t.point_to_local(r.origin()), local_dir});
    bool rv = h->hit(local_ray, scene, isect);
    if (rv) {
        isect.norm = t.vec_from_local(isect.norm);
        isect.time *= dir_len;
    }
    return rv;
}

__device__
bool cast_ray(Scene* scene, const rmath::Ray<float>& r, rprimitives::Isect& isect) {
    // TODO: use bvh tree
    Transformation* trans = scene->get_trans();
    bool hit = false;
    for (int i = 0; i < scene->n_trans(); i++) {
        const Transformation& t = trans[i];
        hit |= cast_local(scene, r, isect, t);
    }
    return hit;
}

__device__
rmath::Vec4<float> propagate_ray(Scene* scene, const rmath::Ray<float>& r) {
    rprimitives::Isect isect;
    enum FrameType {
        NORMAL,
        REFLECT,
        REFRACT
    };

    struct RayFrame {
        rmath::Ray<float> ray;
        rmath::Vec3<float> hit_pt;
        rmath::Vec3<float> norm;
        rmath::Vec4<float> atten;
        rprimitives::Material* last_mat;
        float last_eta;
        FrameType type;
        int depth;
        bool in_obj;
    };
    RayFrame frames[MAX_DEPTH];
    frames[0] = {r, rmath::Vec3<float>(), rmath::Vec3<float>(), rmath::Vec4<float>({1.0f, 1.0f, 1.0f, 1.0f}), 
                        NULL, 1.0f, FrameType::NORMAL, scene->get_recurse_depth(), false};
    int stack_top = 0;
    rmath::Vec4<float> acc_col{};
    
    while (stack_top >= 0) {
        RayFrame& top = frames[stack_top];
        switch (top.type) {
            case FrameType::NORMAL: {
                if (cast_ray(scene, top.ray, isect)) {
                    acc_col += top.atten * rprimitives::illuminate(top.ray, isect, scene);
                    if (top.depth > 0) {
                        if (top.in_obj) {
                            rmath::Vec4<float> kt = top.last_mat->get_Kt();
                            float ar = pow(isect.time, kt[0]);
                            float ag = pow(isect.time, kt[1]);
                            float ab = pow(isect.time, kt[2]);
                            float aa = pow(isect.time, kt[3]);
                            top.atten *= rmath::Vec4<float>({ar, ag, ab, aa});
                        }
                        frames[stack_top].type = FrameType::REFLECT;
                        frames[stack_top].hit_pt = top.ray.at(isect.time);
                        frames[stack_top].last_mat = isect.mat;
                        top.norm = isect.norm;
                    } else {
                        stack_top--;
                    }
                } else {
                    stack_top--;
                }
                break;
            }
            case FrameType::REFLECT: {
                rmath::Vec4<float> kr = isect.mat->get_Kr();
                frames[stack_top].type = FrameType::REFRACT;
                if (kr[0] > 0.0f || kr[1] > 0.0f || kr[2] > 0.0f || kr[3] > 0.0f) {
                    stack_top++;
                    RayFrame& new_top = frames[stack_top];
                    new_top.type = FrameType::NORMAL;
                    new_top.last_mat = top.last_mat;
                    new_top.in_obj = top.in_obj;
                    new_top.atten = top.atten * kr;
                    new_top.depth = top.depth - 1;
                    rmath::Vec3<float> reflect_dir = rmath::reflect(top.ray.direction(), top.norm);
                    new_top.ray = rmath::Ray<float>(top.hit_pt, reflect_dir);
                }
                break;
            }
            case FrameType::REFRACT: {
                rmath::Vec4<float> kt = isect.mat->get_Kt();
                if (kt[0] > 0.0f || kt[1] > 0.0f || kt[2] > 0.0f || kt[3] > 0.0f) {
                    top.type = FrameType::NORMAL;
                    bool tir;
                    rmath::Vec3<float> refract_dir = rmath::refract(top.ray.direction(), top.norm, 
                                        top.last_eta, top.last_mat->get_eta(), tir);
                    if (tir) {
                        stack_top--;
                    } else {
                        top.ray = rmath::Ray<float>(top.hit_pt, refract_dir);
                        top.in_obj = !top.in_obj;
                        top.depth--;
                        if (top.in_obj) {
                            top.last_eta = top.last_mat->get_eta();
                        } else {
                            top.last_eta = 1.0f;
                        }
                    }
                } else {
                    stack_top--;
                }
                break;
            }
        }
    }
    return acc_col;
}
}