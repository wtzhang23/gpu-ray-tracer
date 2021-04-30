#ifndef VERTEX_BUFFER_H
#define VERTEX_BUFFER_H

#include <vector>
#include "gputils/alloc.h"
#include "raymath/linear.h"

namespace rprimitives {
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class VertexBuffer {
private:
    gputils::TextureBuffer4D<float> v;
    gputils::TextureBuffer4D<float> n;
public:
    VertexBuffer(std::vector<rmath::Vec3<float>>& vertices, std::vector<rmath::Vec3<float>>& normals);
    
    CUDA_HOSTDEV
    gputils::TextureBuffer4D<float>& get_vertices() {
        return v;
    }

    CUDA_HOSTDEV
    gputils::TextureBuffer4D<float>& get_normals() {
        return n;
    }
};
}

#endif