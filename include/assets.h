#ifndef GPUTILS_ASSETS_H
#define GPUTILS_ASSETS_H

#include "gputils/alloc.h"

namespace assets {
gputils::TextureBuffer4D<float> read_png(const char* filename);
}

#endif