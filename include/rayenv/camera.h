#ifndef RAYENV_CAMERA_H
#define RAYENV_CAMERA_H

#include "rayprimitives/entity.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {

class Canvas;

class Camera: public rprimitives::Entity {
private:
    float global_near;
    float unit_to_pixels;
    float canvas_width;
    float canvas_height;
public:
    Camera(float fov, float unit_to_pixels, const Canvas& canvas);

    CUDA_HOSTDEV
    rmath::Ray<float> forward() const;

    CUDA_HOSTDEV
    rmath::Ray<float> up() const;

    CUDA_HOSTDEV
    rmath::Ray<float> right() const;

    CUDA_HOSTDEV
    rmath::Ray<float> at(float canvas_x, float canvas_y) const;
};

}

#endif