#include "canvas.h"
#include <cassert>

namespace canvas {
int* create_buffer(int width, int height) {
    int* buffer;
    auto rv = cudaMallocManaged(&buffer, sizeof(int) * width * height);
    assert(rv == 0);
    return buffer;
}

void free_buffer(int* buffer) {
    auto rv = cudaFree(buffer);
    assert(rv == 0);
}
}