#include "rayenv/cpu/scene.h"
#include "raytracer.h"
#include "rayprimitives/isect.h"
#include "rayenv/transformation.h"
#include "rayprimitives/cpu/hitable.h"

namespace rtracer {
namespace cpu {

ropt::cpu::BVH create_bvh(renv::cpu::Scene* scene) {
    renv::Environment& env = scene->get_environment();
    renv::Transformation* trans = env.get_trans();
    std::vector<rprimitives::cpu::Hitable*>& hitables = scene->get_hitables();
    int n_trans = env.n_trans();
    int padded_size = 1 << ((int) ceil(log2(n_trans))); // ensure # of boxes is a power of 2
    std::vector<ropt::BoundingBox> boxes{};
    for (int i = 0; i < n_trans; i++) {
        renv::Transformation& t = trans[i];
        assert(t.get_hitable_idx() < hitables.size());
        rprimitives::cpu::Hitable& h = *hitables[t.get_hitable_idx()];
        boxes.push_back(ropt::from_local(h.compute_bounding_box(scene), t));
    }
    for (int i = n_trans; i < padded_size; i++) {
        boxes.push_back(ropt::BoundingBox());
    }
    return ropt::cpu::BVH(boxes);
}

void debug_cast(renv::cpu::Scene* scene, int x, int y) {
    scene->get_environment().set_debug_mode(true);
    scene->set_bvh(create_bvh(scene));
    renv::Camera& cam = scene->get_environment().get_camera();
    rmath::Ray<float> r = cam.at(x, y);
    float time;
    rprimitives::Isect isect{time};
    rmath::Vec4<float> c = scene->propagate_ray(r, isect);
    scene->set_bvh(ropt::cpu::BVH());
    scene->get_environment().set_debug_mode(false);
}

void update_scene(renv::cpu::Scene* scene, int kernel_dim, bool optimize) {
    renv::Environment& env = scene->get_environment();
    renv::Canvas& canvas = env.get_canvas();
    renv::Camera& cam = env.get_camera();
    rprimitives::cpu::Texture& atlas = scene->get_atlas();
    if (optimize) {
        scene->set_bvh(create_bvh(scene));
    }
    for (int x = 0; x < canvas.get_width(); x++) {
        for (int y = 0; y < canvas.get_height(); y++) {
            rmath::Ray<float> r = cam.at(x, y);
            float time = INFINITY;
            rprimitives::Isect isect{time};
            rmath::Vec4<float> c = scene->propagate_ray(r, isect);
            canvas.set_color(x, y, renv::Color(c[0] > 1.0f ? 1.0f : c[0], 
                                            c[1] > 1.0f ? 1.0f : c[1], 
                                            c[2] > 1.0f ? 1.0f : c[2], 
                                            c[3] > 1.0f ? 1.0f : c[3]));
        }
    }
    scene->set_bvh(ropt::cpu::BVH());
}

}
}