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

struct Shade {
    union Data {
        struct TextData {
            int texture_x;
            int texture_y;
            int texture_width;
            int texture_height;
        } text_data;
        rmath::Vec4<float> col;
        __host__ __device__
        Data(rmath::Vec4<float> col): col(col) {}
        __host__ __device__
        Data(int texture_x, int texture_y, int texture_width, int texture_height): text_data{texture_x, texture_y, texture_width, texture_height}{}
    } data;
    bool use_texture;
    __host__ __device__
    Shade(rmath::Vec4<float> col): data(col), use_texture(false) {}
    __host__ __device__
    Shade(int texture_x, int texture_y, int texture_width, int texture_height): data(texture_x, texture_y, texture_width, texture_height),
                            use_texture(true) {}
};
}

#endif