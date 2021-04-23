#ifndef GPU_ALLOC_H
#define GPU_ALLOC_H

namespace gpu_alloc {
extern void* create_buffer(int n_data, int data_size);
extern void free_buffer(void* buffer);
}

#endif