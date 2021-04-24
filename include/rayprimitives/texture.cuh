#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "rayprimitives/texture.h"

namespace rprimitives {
    template <typename T>
    __device__
    rmath::Vec3<T> get_color_from_texture(const Texture<T>& text, T u, T v) {
        if (text.is_singleton) {
            return text.color.singleton;
        } else {
            const gputils::TextureBuffer3D<T>& texture = text.color.texture;
            int x = u * texture.get_width();
            int y = v * texture.get_height();
            float3 col = tex2D<float3>(texture.get_obj(), y * texture.get_width() + x);
            return rmath::Vec3<T>(col);
        }
    }
}

#endif