#ifndef TEXTURE_CUH
#define TEXTURE_CUH

namespace rprimitives {
class Texture;

__device__
rmath::Vec4<float> get_color_from_texture(Texture& text, float text_x, float text_y);

__device__
rmath::Vec4<float> get_color_from_texture(Texture& text, float u, float v, int text_x, int text_y, int text_width, int text_height);
}

#endif