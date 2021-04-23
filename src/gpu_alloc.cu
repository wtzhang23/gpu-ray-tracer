#include "gpu_alloc.h"
#include <cassert>

namespace gpu_alloc {

void* create_buffer(int n_data, int data_size) {
    void* buffer;
    auto rv = cudaMallocManaged(&buffer, data_size * n_data);
    assert(rv == 0);
    return buffer;
}

void free_buffer(void* buffer) {
    auto rv = cudaFree(buffer);
    assert(rv == 0);
}
}