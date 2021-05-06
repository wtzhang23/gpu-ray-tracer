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
namespace gpu {
class Texture {
private:
    gputils::TextureBuffer4D<float> texture;

    CUDA_HOSTDEV
    static rmath::Vec4<float> color_to_vec(renv::Color c);

    static void free(Texture& texture);
public:
    Texture(gputils::TextureBuffer4D<float> texture): texture(texture) {}

    CUDA_HOSTDEV
    gputils::TextureBuffer4D<float>& get_buffer() {
        return texture;
    }
};
}
}

#endif