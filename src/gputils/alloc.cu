#include "gputils/alloc.h"
#include <cassert>

namespace gputils {
void* create_buffer(const int n_data, const int data_size) {
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