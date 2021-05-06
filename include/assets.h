#ifndef GPUTILS_ASSETS_H
#define GPUTILS_ASSETS_H

#include <vector>
#include "gputils/alloc.h"
#include "raymath/linear.h"
#include "rayprimitives/cpu/texture.h"

namespace assets {

namespace gpu {
gputils::TextureBuffer4D<float> read_png(const char* filename);
}

namespace cpu {
rprimitives::cpu::Texture read_png(const char* filename);
}

}

#endif