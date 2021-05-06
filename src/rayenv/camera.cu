#include "rayenv/camera.h"
#include "rayenv/canvas.h"

namespace renv {

Camera::Camera(float fov, float unit_to_pixels, const Canvas& canvas): rprimitives::Entity(rmath::Vec3<float>(), rmath::Quat<float>::identity()), 
                                                    global_near(0.5f * canvas.get_width() / unit_to_pixels / tan(fov)), 
                                                    unit_to_pixels(unit_to_pixels), canvas_width(canvas.get_width()), 
                                                    canvas_height(canvas.get_height()) {}

CUDA_HOSTDEV
rmath::Ray<float> Camera::forward() const {
    const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
    rmath::Vec3<float> dir = rmath::Vec3<float>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized(); // negative z look direction by convention
    return rmath::Ray<float>(pos(), dir);
}

CUDA_HOSTDEV
rmath::Ray<float> Camera::up() const {
    const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
    rmath::Vec3<float> dir = rmath::Vec3<float>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
    return rmath::Ray<float>(pos(), dir);
}

CUDA_HOSTDEV
rmath::Ray<float> Camera::right() const {
    const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
    rmath::Vec3<float> dir = rmath::Vec3<float>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
    return rmath::Ray<float>(pos(), dir);
}

CUDA_HOSTDEV
rmath::Ray<float> Camera::at(float canvas_x, float canvas_y) const {
    const rmath::Mat3<float> rot_mat = this->o.to_Mat3();
    rmath::Vec3<float> r = rmath::Vec3<float>({rot_mat(0, 0), rot_mat(1, 0), rot_mat(2, 0)}).normalized();
    rmath::Vec3<float> u = rmath::Vec3<float>({rot_mat(0, 1), rot_mat(1, 1), rot_mat(2, 1)}).normalized();
    rmath::Vec3<float> f = rmath::Vec3<float>({rot_mat(0, 2), rot_mat(1, 2), rot_mat(2, 2)}).normalized();
    float global_x = (canvas_x - (0.5f * canvas_width)) / unit_to_pixels;
    float global_y = (0.5f * canvas_height - canvas_y) / unit_to_pixels;
    rmath::Vec3<float> dir = global_near * f + global_x * r + global_y * u; 
    return rmath::Ray<float>(pos(), dir);
}

}