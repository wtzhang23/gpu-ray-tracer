#ifndef RAYPRIMITIVES_GPU_TEXTURE_CUH
#define RAYPRIMITIVES_GPU_TEXTURE_CUH

namespace rprimitives {
namespace gpu {
class Texture;
__device__
rmath::Vec4<float> get_color_from_texture(Texture& text, float text_x, float text_y);
}
}

#endif