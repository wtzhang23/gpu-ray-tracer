#ifndef GPU_ALLOC_H
#define GPU_ALLOC_H

namespace gputils {
extern void* create_buffer(const int n_data, const int data_size);
extern void free_buffer(void* buffer);
}

#endif