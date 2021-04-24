#ifndef TEXTURE_H
#define TEXTURE_H

#include "raymath/linear.h"
#include "rayenv/color.h"
#include "gputils/flat_vec.h"
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
    const gputils::FlatVec<T, 3> texture;
    const int width;
    const int height;

    static rmath::Vec3<T> color_to_vec(renv::Color c) {
        return (T) 1 / UINT8_MAX * rmath::Vec3<T>((T) c.r(), (T) c.g(), (T) c.b());
    }

public:
    Texture(renv::Color** texture, int width, int height) {
        // TODO
    }

    Texture(renv::Color color): texture(), width(0), height(0), color(color){}

    Texture(): Texture(renv::Color(1.0f, 1.0f, 1.0f, 1.0f)){}

    CUDA_HOSTDEV
    rmath::Vec3<T> get_color(T u, T v) {
        if (texture.size() == 0) {
            return color;
        } else {
            int x = u * width;
            int y = v * height;
            T col[3] = texture.get_vec(y * width + x);
            return rmath::Vec3<T>(col);
        }
    }
};
}

#endif