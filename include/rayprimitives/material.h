#ifndef MATERIAL_H
#define MATERIAL_H

#include "raymath/linear.h"
#include "rayprimitives/texture.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
class Material {
private:
    rmath::Vec3<float> Ke;
    rmath::Vec3<float> Kd;
    rmath::Vec3<float> Ks;    
    rmath::Vec3<float> Kt;
    rmath::Vec3<float> Kr;
    float alpha;
    float eta;
public:
    CUDA_HOSTDEV
    Material(rmath::Vec3<float> Ke, rmath::Vec3<float> Kd, rmath::Vec3<float> Ks, rmath::Vec3<float> Kt, rmath::Vec3<float> Kr,
                float alpha, float eta): Ke(Ke), Kd(Kd), Ks(Ks), Kt(Kt), Kr(Kr), alpha(alpha), eta(eta){}
    
    CUDA_HOSTDEV
    Material(): Material(rmath::Vec3<float>({1.0f, 1.0f, 1.0f}), rmath::Vec3<float>(), rmath::Vec3<float>(),
                rmath::Vec3<float>(), rmath::Vec3<float>(), 0, 0){}

};
}

#endif