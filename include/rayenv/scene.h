#ifndef SCENE_H
#define SCENE_H

#include <array>
#include <vector>
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayenv/canvas.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/texture.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {
class Camera: private rprimitives::Entity {
private:
    float near;
    int width;
    int height;
public:
    Camera(float fov, const Canvas& canvas): rprimitives::Entity(rmath::Vec3<float>(), rmath::Quat<float>::identity()), near(0.5f * canvas.get_width() / tan(fov)), 
                                                    width(canvas.get_width()), height(canvas.get_height()) {}

    CUDA_HOSTDEV
    rmath::Ray<float> forward() const {
        const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
        rmath::Vec3<float> dir = -rmath::Vec3<float>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized(); // negative z look direction by convention
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
    std::array<rmath::Vec3<float>, 3> basis() const {
        const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
        rmath::Vec3<float> r = rmath::Vec3<float>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        rmath::Vec3<float> u = rmath::Vec3<float>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        rmath::Vec3<float> f = -rmath::Vec3<float>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized();
        return {r, u, f};
    }

    CUDA_HOSTDEV
    rmath::Ray<float> at(float x, float y) const  {
        std::array<rmath::Vec3<float>, 3> b = basis();
        rmath::Vec3<float> dir = near * b[2] + (x - 0.5f * width) * b[0] + (y - 0.5f * height) * b[1]; 
        return rmath::Ray<float>(pos(), dir);
    }
};

class Scene {
private:
    Canvas canvas;
    Camera cam;
    rprimitives::Texture atlas;
    rprimitives::Hitable** hitables;
    int n_hitables;
public:
    Scene(Canvas& canvas, Camera camera, rprimitives::Texture atlas, std::vector<rprimitives::Hitable*> hitables);

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
};
}
#endif