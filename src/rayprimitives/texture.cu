#include "rayprimitives/texture.h"
#include "rayprimitives/texture.cuh"
#include "raymath/linear.h"

namespace rprimitives{
__device__
rmath::Vec4<float> get_color_from_texture(Texture& text, float u, float v, int text_x, int text_y, int text_width, int text_height) {
    gputils::TextureBuffer4D<float>& texture = text.get_buffer();
    float offset_x = u * text_width;
    float offset_y = v * text_height;
    float4 col = tex2D<float4>(texture.get_obj(), text_x + offset_x, text_y + offset_y);
    return rmath::Vec4<float>({col.x, col.y, col.z, col.w});
}

rmath::Vec4<float> Texture::color_to_vec(renv::Color c) {
    return 1.0f / UINT8_MAX * rmath::Vec4<float>({(float) c.r(), (float) c.g(), (float) c.b(), (float) c.a()});
}

void Texture::free(Texture& texture) {
    gputils::TextureBuffer4D<float>::free(texture.get_buffer());
}
}