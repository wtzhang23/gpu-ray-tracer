#ifndef TEXTURE_H
#define TEXTURE_H

#include "raymath/linear.h"
#include "rayenv/color.h"
#include "gputils/alloc.h"
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
    const union TexInner {
        rmath::Vec3<T> singleton;
        gputils::TextureBuffer3D<T> texture;
        TexInner(rmath::Vec3<T> singleton): singleton(singleton){}
        TexInner(gputils::TextureBuffer3D<T> texture): texture(texture){}
    } color;
    bool is_singleton;

    static rmath::Vec3<T> color_to_vec(renv::Color c) {
        return (T) 1 / UINT8_MAX * rmath::Vec3<T>((T) c.r(), (T) c.g(), (T) c.b());
    }

public:
    Texture(gputils::TextureBuffer3D<T> texture): is_singleton(false), color(texture) {
        // TODO
    }

    Texture(renv::Color color): is_singleton(true), color(color_to_vec(color)){}

    Texture(): Texture(renv::Color(1.0f, 1.0f, 1.0f, 1.0f)){}
};
}

#endif