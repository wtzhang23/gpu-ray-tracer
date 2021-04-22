#ifndef SCENE_H
#define SCENE_H

#include <array>
#include "linear.h"
#include "geometry.h"
#include "canvas.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace scene {
template <typename T>
class Camera {
private:
    geometry::Quat<T> o;
    linear::Vec3<T> p;
    T near;
    int width;
    int height;
public:
    Camera(T fov, const canvas::Canvas& canvas): o(geometry::Quat<T>::identity()), p(), near(0.5 * canvas.get_width() / tan(fov)), 
                                                    width(canvas.get_width()), height(canvas.get_height()) {}

    linear::Vec3<T> pos() const {
        return p;
    }

    geometry::Ray<T> forward() const {
        const linear::Mat3<T> rot_mat = o.to_Mat3();
        linear::Vec3<T> dir = -linear::Vec3<T>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized(); // negative z look direction by convention
        return geometry::Ray<T>(o, dir);
    }

    geometry::Ray<T> up() const {
        const linear::Mat3<T> rot_mat = o.to_Mat3();
        linear::Vec3<T> dir = linear::Vec3<T>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        return geometry::Ray<T>(o, dir);
    }

    geometry::Ray<T> right() const {
        const linear::Mat3<T> rot_mat = o.to_Mat3();
        linear::Vec3<T> dir = linear::Vec3<T>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        return geometry::Ray<T>(o, dir);
    }

    std::array<geometry::Ray<T>, 3> basis() const {
        const linear::Mat3<T> rot_mat = o.to_Mat3();
        linear::Vec3<T> r = linear::Vec3<T>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
        linear::Vec3<T> u = linear::Vec3<T>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
        linear::Vec3<T> f = -linear::Vec3<T>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized();
        return {r, u, f};
    }

    geometry::Ray<T> at(T x, T y) const  {
        geometry::Ray<T> b[3] = basis();
        linear::Vec3<T> dir = near * b[2] + (x - 0.5 * width) * b[0] + (y - 0.5 * height) * b[1]; 
    }

    void translate_global(linear::Vec3<T> dp) {
        p += dp;
    }

    void translate(linear::Vec3<T> dp) {
        translate_global(o.to_Mat3() * dp);
    }

    void rotate(geometry::Quat<T> dr) {
        o = dr * o;
    }

    void set_position(linear::Vec3<T> p) {
        this->p = p;
    }

    void set_orientation(geometry::Quat<T> o) {
        this->o = o;
    }
};

template <typename T>
class Scene {
private:
    canvas::Canvas canvas;
    Camera<T> cam;
public:
    Scene(canvas::Canvas& canvas, Camera<T> camera): canvas(canvas), cam(camera){}

    CUDA_HOSTDEV
    Scene(const Scene&) = default;

    CUDA_HOSTDEV
    canvas::Canvas& get_canvas() {
        return canvas;
    }

    CUDA_HOSTDEV
    Camera<T>& get_camera() {
        return cam;
    }
};
}
#endif