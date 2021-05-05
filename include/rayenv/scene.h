#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayenv/canvas.h"
#include "rayopt/bvh.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/texture.h"
#include "rayprimitives/vertex_buffer.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
class Hitable;
class Light;
}

namespace renv {
class Camera: public rprimitives::Entity {
private:
    float global_near;
    float unit_to_pixels;
    float canvas_width;
    float canvas_height;
public:
    Camera(float fov, float unit_to_pixels, const Canvas& canvas): rprimitives::Entity(rmath::Vec3<float>(), rmath::Quat<float>::identity()), 
                                                    global_near(0.5f * canvas.get_width() / unit_to_pixels / tan(fov)), 
                                                    unit_to_pixels(unit_to_pixels), canvas_width(canvas.get_width()), 
                                                    canvas_height(canvas.get_height()) {}

    CUDA_HOSTDEV
    rmath::Ray<float> forward() const {
        const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
        rmath::Vec3<float> dir = rmath::Vec3<float>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized(); // negative z look direction by convention
        return rmath::Ray<float>(pos(), dir);
    }

    CUDA_HOSTDEV
    rmath::Ray<float> up() const {
        const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
        rmath::Vec3<float> dir = rmath::Vec3<float>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        return rmath::Ray<float>(pos(), dir);
    }

    CUDA_HOSTDEV
    rmath::Ray<float> right() const {
        const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
        rmath::Vec3<float> dir = rmath::Vec3<float>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        return rmath::Ray<float>(pos(), dir);
    }

    CUDA_HOSTDEV
    rmath::Ray<float> at(float canvas_x, float canvas_y) const {
        const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
        rmath::Vec3<float> r = rmath::Vec3<float>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        rmath::Vec3<float> u = rmath::Vec3<float>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        rmath::Vec3<float> f = rmath::Vec3<float>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized();
        float global_x = (canvas_x - (0.5f * canvas_width)) / unit_to_pixels;
        float global_y = (0.5f * canvas_height - canvas_y) / unit_to_pixels;
        rmath::Vec3<float> dir = global_near * f + global_x * r + global_y * u; 
        return rmath::Ray<float>(pos(), dir);
    }
};

class Scene;

class Transformation: public rprimitives::Entity {
private:
    int hitable_idx;
public:
    Transformation(int hi): hitable_idx(hi) {}
    CUDA_HOSTDEV
    int get_hitable_idx() const {
        return hitable_idx;
    }
};

class Scene {
private:
    Canvas canvas;
    Camera cam;
    rprimitives::Texture atlas;
    rprimitives::Hitable** hitables;
    rprimitives::Light** lights;
    Transformation* trans;
    rmath::Vec3<float> dist_atten;
    rmath::Vec4<float> ambience;
    rprimitives::VertexBuffer buffer;
    ropt::BVH bvh;
    int nh;
    int nl;
    int nt;
    int recurse_depth;
    bool debugging;
public:
    Scene(Canvas canvas, Camera camera, rprimitives::Texture atlas, 
                rprimitives::Hitable** hitables, int n_hitables, 
                rprimitives::Light** lights, int n_lights,
                Transformation* trans, int n_trans,
                rprimitives::VertexBuffer buffer): canvas(canvas), cam(camera),
                atlas(atlas), hitables(hitables), lights(lights), trans(trans),
                dist_atten(), ambience(), buffer(buffer), bvh(), nh(n_hitables), 
                nl(n_lights), nt(n_trans), recurse_depth(0), debugging(false) {}

    void set_dist_atten(float const_term, float linear_term, float quad_term) {
        dist_atten = rmath::Vec3<float>({const_term, linear_term, quad_term});
    }

    void set_ambience(rmath::Vec4<float> amb) {
        ambience = amb;
    }

    void set_recurse_depth(int depth) {
        recurse_depth = depth;
    }

    void set_debug_mode(bool mode) {
        debugging = mode;
    }

    void set_bvh(ropt::BVH bvh) {
        this->bvh = bvh;
    }

    CUDA_HOSTDEV
    const ropt::BVH& get_bvh() const {
        return bvh;
    }

    CUDA_HOSTDEV
    bool is_debugging() const {
        return debugging;
    }

    CUDA_HOSTDEV
    const rmath::Vec3<float>& get_dist_atten() const {
        return dist_atten;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_ambience() const {
        return ambience;
    }

    CUDA_HOSTDEV
    int get_recurse_depth() const {
        return recurse_depth;
    }

    CUDA_HOSTDEV
    Canvas& get_canvas() {
        return canvas;
    }

    CUDA_HOSTDEV
    Camera& get_camera() {
        return cam;
    }

    CUDA_HOSTDEV
    rprimitives::Texture& get_atlas() {
        return atlas;
    }

    CUDA_HOSTDEV
    rprimitives::Hitable** get_hitables() {
        return hitables;
    }

    CUDA_HOSTDEV
    Transformation* get_trans() {
        return trans;
    }

    CUDA_HOSTDEV
    rprimitives::Light** get_lights() {
        return lights;
    }

    CUDA_HOSTDEV
    rprimitives::VertexBuffer& get_vertex_buffer() {
        return buffer;
    }

    CUDA_HOSTDEV
    int n_hitables() const {
        return nh;
    }

    CUDA_HOSTDEV
    int n_lights() const {
        return nl;
    }

    CUDA_HOSTDEV
    int n_trans() const {
        return nt;
    }
};
}
#endif