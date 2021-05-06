#ifndef RAYPRIMITIVES_ISECT_H
#define RAYPRIMITIVES_ISECT_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {

class TextureCoords;
class Material;

struct Isect {
    float& time;
    rmath::Vec3<float> norm;
    rmath::Vec<float, 2> uv;
    TextureCoords* coords;
    Material* mat;

    CUDA_HOSTDEV
    Isect(float& time): time(time){}
};

}

#endif