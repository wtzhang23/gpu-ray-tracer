#include "rayprimitives/gpu/texture.h"
#include "rayprimitives/gpu/texture.cuh"
#include "raymath/linear.h"

namespace rprimitives {
namespace gpu {

__device__
rmath::Vec4<float> get_color_from_texture(Texture& text, float text_x, float text_y) {
    float4 col = tex2D<float4>(text.get_buffer().get_obj(), text_x, text_y);
    return rmath::Vec4<float>({col.x, col.y, col.z, col.w});
}

rmath::Vec4<float> Texture::color_to_vec(renv::Color c) {
    return 1.0f / UINT8_MAX * rmath::Vec4<float>({(float) c.r(), (float) c.g(), (float) c.b(), (float) c.a()});
}

void Texture::free(Texture& texture) {
    gputils::TextureBuffer4D<float>::free(texture.get_buffer());
}

}
}