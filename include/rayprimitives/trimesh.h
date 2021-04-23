#ifndef TRIMESH_H
#define TRIMESH_H

#include "rayprimitives/primitive.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace primitives {
template<typename T>
class Trimesh {
private:
    int n_triangles;
    int n_vertices;
    T* vertex_x;
    T* vertex_y;
    T* vertex_z;
    T* normal_x;
    T* normal_y;
    T* normal_z;
    int* vertex_a;
    int* vertex_b;
    int* vertex_c;
public:
    Trimesh(int n_triangles) {}

    CUDA_HOSTDEV
    ~Trimesh() {}
};
}

#endif