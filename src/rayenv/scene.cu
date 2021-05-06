#include "rayenv/gpu/scene.h"
#include "rayenv/gpu/scene.cuh"
#include "rayenv/cpu/scene.h"
#include "rayenv/transformation.h"
#include "rayprimitives/material.h"
#include "rayprimitives/isect.h"
#include "rayprimitives/gpu/phong.cuh"
#include "rayprimitives/gpu/hitable.cuh"
#include "rayprimitives/cpu/phong.h"
#include "rayprimitives/cpu/hitable.h"
#include "rayopt/bounding_box.h"

namespace renv {
__host__ __device__
rmath::Vec4<float> trans_atten(const rprimitives::Material& mat, float time) {
    rmath::Vec4<float> kt = mat.get_Kt();
    float ar = pow(time, kt[0]);
    float ag = pow(time, kt[1]);
    float ab = pow(time, kt[2]);
    float aa = pow(time, kt[3]);
    return rmath::Vec4<float>({ar, ag, ab, aa});
}

namespace gpu {
static const int MAX_DEPTH = 10;

__device__
bool cast_local(Scene* scene, const rmath::Ray<float>& r, rprimitives::Isect& isect, const Transformation& t) {
    rprimitives::gpu::Hitable** hitables = scene->get_hitables();
    rprimitives::gpu::Hitable* h = hitables[t.get_hitable_idx()];
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
    Environment& env = scene->get_environment();
    Transformation* trans = env.get_trans();
    bool hit = false;
    const ropt::gpu::BVH& bvh = scene->get_bvh();
    if (bvh.empty()) {
        for (int i = 0; i < env.n_trans(); i++) {
            const Transformation& t = trans[i];
            hit |= cast_local(scene, r, isect, t);
        }
    } else {
        ropt::gpu::BVHIterator iter{r, INFINITY, scene};
        while (iter.current() >= 0) {
            const Transformation& t = trans[iter.current()];
            if (cast_local(scene, r, isect, t)) {
                hit = true;
            }
            iter.next(INFINITY);
        }
        if (env.is_debugging()) {
            printf("tested %d / %d bounding boxes for %d objs\n", 
                    iter.n_intersections(), iter.max_intersections(), env.n_trans());
        }
    }
    return hit;
}

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
    const rprimitives::Material* last_mat;
    FrameType type;
    int depth;
    bool in_obj;
};

__device__
rmath::Vec4<float> propagate_ray(Scene* scene, const rmath::Ray<float>& r, rprimitives::Isect& isect) {
    renv::Environment& env = scene->get_environment();

    RayFrame frames[MAX_DEPTH];
    frames[0] = {r, rmath::Vec3<float>(), rmath::Vec3<float>(), rmath::Vec4<float>({1.0f, 1.0f, 1.0f, 1.0f}), 
                        NULL, FrameType::NORMAL, env.get_recurse_depth(), false};
    int stack_top = 0;
    rmath::Vec4<float> acc_col{};
    
    while (stack_top >= 0) {
        RayFrame& top = frames[stack_top];
        switch (top.type) {
            case FrameType::NORMAL: {
                isect.time = INFINITY; // reset
                if (env.is_debugging()) {
                    printf("shooting a ray\n");
                }
                if (cast_ray(scene, top.ray, isect)) {
                    if (top.depth > 0) {
                        if (top.in_obj) {
                            assert(top.last_mat->refractive());
                            top.atten *= renv::trans_atten(*isect.mat, isect.time);
                        }
                        frames[stack_top].type = FrameType::REFLECT;
                        frames[stack_top].hit_pt = top.ray.at(isect.time);
                        frames[stack_top].last_mat = isect.mat;
                        top.norm = isect.norm;
                    } else {
                        stack_top--;
                    }

                    acc_col += top.atten * rprimitives::gpu::illuminate(top.ray, isect, scene);
                } else {
                    stack_top--;
                }
                break;
            }
            case FrameType::REFLECT: {
                rmath::Vec4<float> kr = isect.mat->get_Kr();
                frames[stack_top].type = FrameType::REFRACT;
                if (isect.mat->reflective()) {
                    if (env.is_debugging()) {
                        printf("preparing to shoot a reflection ray\n");
                    }
                    stack_top++;
                    RayFrame& new_top = frames[stack_top];
                    new_top.type = FrameType::NORMAL;
                    new_top.last_mat = top.last_mat;
                    new_top.in_obj = top.in_obj;
                    new_top.atten = top.atten * kr;
                    new_top.depth = top.depth - 1;
                    rmath::Vec3<float> reflect_dir = rmath::reflect(top.ray.direction(), top.norm.normalized());
                    new_top.ray = rmath::Ray<float>(top.hit_pt, reflect_dir);
                }
                break;
            }
            case FrameType::REFRACT: {
                rmath::Vec4<float> kt = isect.mat->get_Kt();
                if (isect.mat->refractive()) {
                    if (env.is_debugging()) {
                        printf("preparing to shoot a refraction ray\n");
                    }
                    top.type = FrameType::NORMAL;
                    

                    float n1;
                    float n2;
                    bool tir;

                    // calculate refraction indices
                    if (top.in_obj) {
                        n1 = top.last_mat->get_eta();
                        n2 = 1.0f;
                    } else {
                        n1 = 1.0f;
                        n2 = top.last_mat->get_eta();
                    }

                    rmath::Vec3<float> refract_dir = rmath::refract(top.ray.direction(), top.norm.normalized(), 
                                        n1, n2, tir);
                    if (tir) {
                        stack_top--;
                    } else {
                        top.ray = rmath::Ray<float>(top.hit_pt, refract_dir);
                        top.in_obj = !top.in_obj;
                        top.depth--;
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

namespace cpu {
    bool Scene::cast_local(const rmath::Ray<float>& r, rprimitives::Isect& isect, const Transformation& t) {
        const rprimitives::cpu::Hitable& h = *hitables[t.get_hitable_idx()];
        rmath::Vec3<float> local_dir = t.vec_to_local(r.direction());
        float dir_len = local_dir.len();
        rmath::Ray<float> local_ray = rmath::Ray<float>({t.point_to_local(r.origin()), local_dir});
        bool rv = h.hit(local_ray, this, isect);
        if (rv) {
            isect.norm = t.vec_from_local(isect.norm);
            isect.time *= dir_len;
        }
        return rv;
    }
    
    bool Scene::cast_ray(const rmath::Ray<float>& r, rprimitives::Isect& isect) {
        Transformation* trans = env.get_trans();
        bool hit = false;
        if (bvh.empty()) {
            for (int i = 0; i < env.n_trans(); i++) {
                const Transformation& t = trans[i];
                hit |= cast_local(r, isect, t);
            }
        } else {
            bvh.traverse(r, [&](int tid) {
                hit |= cast_local(r, isect, trans[tid]);
            });
        }
        
        return hit;
    }

    rmath::Vec4<float> Scene::propagate_helper(const rmath::Ray<float>& r, rprimitives::Isect& isect, 
                                                                           float& last_time, bool in_obj, int depth) {
        if (depth > env.get_recurse_depth()) {
            return rmath::Vec4<float>();
        }
        isect.time = INFINITY; // reset
        if (env.is_debugging()) {
            printf("shooting a ray\n");
        }
        if (cast_ray(r, isect)) {
            rmath::Vec4<float> acc_col = rprimitives::cpu::illuminate(r, isect, this);
            last_time = isect.time;

            float next_time = 0;
            if (isect.mat->reflective()) {
                rmath::Vec3<float> reflect_dir = rmath::reflect(r.direction(), isect.norm);
                rmath::Ray<float> reflect_ray = rmath::Ray<float>(r.at(isect.time), reflect_dir);
                acc_col += propagate_helper(reflect_ray, isect, next_time, in_obj, depth - 1);
            }

            if (isect.mat->refractive()) {
                float n1;
                float n2;
                bool tir;

                // calculate indexes of refraction
                if (in_obj) {
                    n1 = isect.mat->get_eta();
                    n2 = 1.0f;
                } else {
                    n1 = 1.0f;
                    n2 = isect.mat->get_eta();
                }

                rmath::Vec3<float> refract_dir = rmath::refract(r.direction(), isect.norm, n1, n2, tir);
                if (!tir) {
                    next_time = 0;
                    rmath::Ray<float> reflect_ray = rmath::Ray<float>(r.at(isect.time), refract_dir);
                    rmath::Vec4<float> unattenuated_col = propagate_helper(reflect_ray, isect, next_time, in_obj, depth - 1);
                    acc_col += trans_atten(*isect.mat, next_time) * unattenuated_col;
                }
            }
            return acc_col;
        } else {
            return rmath::Vec4<float>();
        }
    }

    rmath::Vec4<float> Scene::propagate_ray(const rmath::Ray<float>& r, rprimitives::Isect& isect) {
        float t;
        return propagate_helper(r, isect, t, false, env.get_recurse_depth());
    }
}
}