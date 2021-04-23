#ifndef SCENE_H
#define SCENE_H

#include <array>
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayenv/canvas.h"
#include "entity.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {
template <typename T>
class Camera: private rentity::Entity<T> {
private:
    T near;
    int width;
    int height;
public:
    Camera(T fov, const Canvas& canvas): rentity::Entity<T>(rmath::Vec3<T>(), rmath::Quat<T>::identity()), near(0.5 * canvas.get_width() / tan(fov)), 
                                                    width(canvas.get_width()), height(canvas.get_height()) {}

    rmath::Vec3<T> pos() const {
        return this->p;
    }

    rmath::Ray<T> forward() const {
        const rmath::Mat3<T> rot_mat = this->o.to_Mat3();
        rmath::Vec3<T> dir = -rmath::Vec3<T>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized(); // negative z look direction by convention
        return rmath::Ray<T>(this->o, dir);
    }

    rmath::Ray<T> up() const {
        const rmath::Mat3<T> rot_mat = this->o.to_Mat3();
        rmath::Vec3<T> dir = rmath::Vec3<T>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        return rmath::Ray<T>(this->o, dir);
    }

    rmath::Ray<T> right() const {
        const rmath::Mat3<T> rot_mat = this->o.to_Mat3();
        rmath::Vec3<T> dir = rmath::Vec3<T>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        return rmath::Ray<T>(this->o, dir);
    }

    std::array<rmath::Ray<T>, 3> basis() const {
        const rmath::Mat3<T> rot_mat = this->o.to_Mat3();
        rmath::Vec3<T> r = rmath::Vec3<T>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        rmath::Vec3<T> u = rmath::Vec3<T>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        rmath::Vec3<T> f = -rmath::Vec3<T>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized();
        return {r, u, f};
    }

    rmath::Ray<T> at(T x, T y) const  {
        rmath::Ray<T> b[3] = basis();
        rmath::Vec3<T> dir = near * b[2] + (x - 0.5f * width) * b[0] + (y - 0.5f * height) * b[1]; 
    }
};

template <typename T>
class Scene {
private:
    Canvas canvas;
    Camera<T> cam;
public:
    Scene(Canvas& canvas, Camera<T> camera): canvas(canvas), cam(camera){}

    CUDA_HOSTDEV
    Scene(const Scene&) = default;

    CUDA_HOSTDEV
    Canvas& get_canvas() {
        return canvas;
    }

    CUDA_HOSTDEV
    Camera<T>& get_camera() {
        return cam;
    }
};
}
#endif