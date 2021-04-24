#ifndef GPU_ALLOC_H
#define GPU_ALLOC_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cuda_runtime.h>

namespace gputils {

template<typename T, int Dim = 1>
class TextureBuffer {
private:
    cudaTextureObject_t obj;
    cudaArray_t cu_array;
    int height;
    int width;

public:
    TextureBuffer<T, Dim>(T* buffer, int width, int height): obj(), cu_array(), height(height), width(width) {
        int y_chn_bits;
        if (Dim > 1) {
            y_chn_bits = sizeof(T) * 8;
        } else {
            y_chn_bits = 0;
        }
        int z_chn_bits;
        if (Dim > 2) {
            z_chn_bits = sizeof(T) * 8;
        } else {
            z_chn_bits = 0;
        }
        int w_chn_bits;
        if (Dim > 3) {
            w_chn_bits = sizeof(T) * 8;
        } else {
            w_chn_bits = 0;
        }

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(T) * 8, y_chn_bits, z_chn_bits, 
                                                                                    w_chn_bits, cudaChannelFormatKindFloat);
        cudaMallocArray(&cu_array, &channelDesc, width, height);
        int memcpyHeight;
        if (height == 0) {
            memcpyHeight = 1;
        } else {
            memcpyHeight = height;
        }
        cudaMemcpy2DToArray(cu_array, 0, 0, buffer, sizeof(T) * width, sizeof(T) * height, memcpyHeight, cudaMemcpyHostToDevice);
        
        cudaResourceDesc resDesc = {};
        resDesc.res.array.array = cu_array;
        resDesc.resType = cudaResourceTypeArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;

        cudaCreateTextureObject(&obj, &resDesc, &texDesc, NULL);
    }
};
extern void* create_buffer(const int n_data, const int data_size);
extern void free_buffer(void* buffer);

template <typename T, int Dim>
void free_texture_buffer(TextureBuffer<T, Dim>& buf) {
    cudaDestroyTextureObject(buf.obj);
    cudaFreeArray(buf.cu_array);
}

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