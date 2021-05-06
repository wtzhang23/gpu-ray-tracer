#ifndef GPUTILS_ALLOC_H
#define GPUTILS_ALLOC_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cuda_runtime.h>

namespace gputils {

template<typename T>
T* copy_to_gpu(T* cpu, int n) {
    T* rv = NULL;
    if (n > 0) {
        cudaMalloc(&rv, sizeof(T) * n);
        cudaMemcpy(rv, cpu, sizeof(T) * n, cudaMemcpyHostToDevice);
    }
    return rv;
}

template<typename T, int Dim = 1>
class TextureBuffer {
private:
    cudaTextureObject_t obj;
    cudaArray_t cu_array;
    int height;
    int width;
public:
    TextureBuffer<T, Dim>(T* buffer, int width, int height): obj(), cu_array(), height(height), width(width) {
        int y_chn_bits = Dim > 1 ? sizeof(T) * 8 : 0;
        int z_chn_bits = Dim > 2 ? sizeof(T) * 8 : 0;
        int w_chn_bits = Dim > 3 ? sizeof(T) * 8 : 0;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(T) * 8, y_chn_bits, z_chn_bits, 
                                                                                    w_chn_bits, cudaChannelFormatKindFloat);
        cudaMallocArray(&cu_array, &channelDesc, width * Dim, height * Dim);
        int memcpyHeight;
        if (height == 0) {
            memcpyHeight = 1;
        } else {
            memcpyHeight = height;
        }
        cudaMemcpy2DToArray(cu_array, 0, 0, buffer, width * Dim * sizeof(T), width * Dim * sizeof(T), memcpyHeight, cudaMemcpyHostToDevice);
        
        cudaResourceDesc resDesc = {};
        resDesc.res.array.array = cu_array;
        resDesc.resType = cudaResourceTypeArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;

        cudaCreateTextureObject(&obj, &resDesc, &texDesc, NULL);
    }

    static void free(TextureBuffer<T, Dim>& buf) {
        cudaDestroyTextureObject(buf.obj);
        cudaFreeArray(buf.cu_array);
    }

    CUDA_HOSTDEV
    cudaTextureObject_t get_obj() {
        return obj;
    }

    CUDA_HOSTDEV
    int get_width() const {
        return width;
    }

    CUDA_HOSTDEV
    int get_height() const {
        return height;
    }
};
extern void* create_buffer(const int n_data, const int data_size);
extern void free_buffer(void* buffer);

template <typename T>
using TextureBuffer1D = TextureBuffer<T, 1>;

template <typename T>
using TextureBuffer2D = TextureBuffer<T, 2>;

template <typename T>
using TextureBuffer3D = TextureBuffer<T, 3>;

template <typename T>
using TextureBuffer4D = TextureBuffer<T, 4>;
}

#endif