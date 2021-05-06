#ifndef RAYPRIMITIVES_CPU_TEXTURE_H
#define RAYPRIMITIVES_CPU_TEXTURE_H

#include <vector>
#include "raymath/linear.h"
#include "rayenv/color.h"

namespace rprimitives {
namespace cpu {
class Texture {
private:
    std::vector<rmath::Vec4<float>> colors;
    int height;
    int width;
public:
    Texture(std::vector<rmath::Vec4<float>> colors, int width, int height): 
                                        colors(colors), height(height), width(width){}
    
    const rmath::Vec4<float>& get_color(int x, int y) const {
        return colors[x + y * width];
    }
};
}
}

#endif