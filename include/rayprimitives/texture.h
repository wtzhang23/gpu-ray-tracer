#ifndef TEXTURE_H
#define TEXTURE_H

#include "raymath/linear.h"
#include "rayenv/color.h"
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
template <typename T>
class Texture {
private:
    const rmath::Vec3<T> color;
    const T* texture_x;
    const T* texture_y;
    const T* texture_z;
    const int width;
    const int height;
    bool degenerate;

    static rmath::Vec3<T> color_to_vec(renv::Color c) {
        return (T) 1 / UINT8_MAX * rmath::Vec3<T>((T) c.r(), (T) c.g(), (T) c.b());
    }

public:
    Texture(renv::Color** texture, int width, int height) {
        // TODO
    }

    Texture(renv::Color color): texture_x(NULL), texture_y(NULL), texture_z(NULL), width(0), height(0), color(color), degenerate(true){}

    Texture(): Texture(renv::Color(1.0f, 1.0f, 1.0f, 1.0f)){}

    CUDA_HOSTDEV
    rmath::Vec3<T> get_color(T u, T v) {
        if (degenerate) {
            return color;
        } else {
            int x = u * width;
            int y = v * height;
            return rmath::Vec3<T>(texture_x[y * width + x], texture_y[y * width + x], texture_z[y * width + x]);
        }
    }
};
}

#endif