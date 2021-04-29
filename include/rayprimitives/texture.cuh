#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "rayprimitives/texture.h"

namespace rprimitives {
    __device__
    rmath::Vec4<float> get_color_from_texture(Texture& text, float u, float v, int text_x, int text_y, int text_width, int text_height);
}

#endif